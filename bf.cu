// High level notes
//   Frankly speaking, simulation of a language is not the best suited to gpu execution and comes with some serious considerations:
//     The completely generic control flow is fundamentally a loop in which each iteration is dependent on the last
//     This alone precludes many optimizations which are typically desirable; with how generic things are the loop cannot be parallelized (save _maybe_ certain explicitly defined special cases)
//     Since we do want to have a maximum execution time per program, the loop length does indeed have an upper bound, which, if structured appropriately, could afford some optimization in the way of loop unrolling
//     However, on the surface of it this may not be compatible with conditional short-circuiting of the loop which itself is desirable enough to probably not be worth giving up (many programs will end very quickly)
//     That said, highly controlled partial loop unrollment (essentially implemented as a nested loop which runs a constant number of times) is also an option which is worth exploring,
//     the inner loop would essentially no-op on each loop execution after it has short-circuted.
//     I suppose it is possible that cuda would try something like this anyway, and while I find it unlikely that it is smart enough to figure it out in this case it should be investigated before attempting alternatives
//
//     An interesting (crazy) thought is to attempt to directly do speculative execution and/or branch prediction with the bf program's control flow.
//     This may not be as intractable as it first seems due to the fact that the control flow is both quite simple and represented directly (all cases are a zero/non-zero jump),
//     as well as the fact that the theoretical pipelining _should_ be somewhat straightforward
//
//     Another thought is to actually directly compile the bf programs to gpu assembly before execution. My intuition is that this per-program compilation overhead would quickly overwhelm the increased "runtime" optimization
//     if this were to be done the compilation would have to be extremely simple, probably close to 1-to-1
//     That said, bf is simple enough that a somewhat 1-to-1 compilation would theoretically be done quite quickly, so depending on how fast the gpu assembly can emit this _maybe_ could work
//     Another huge confounder is that this would require seperate executable memory to be uploaded per thread/block, I have no idea if that is even possible without multiple host calls
//     I suppose to the last point, this could also be done with regular cpu assembly or an intermediate like llvm, which might be an interesting avenue
//
//   Having a good profiling baseline will be very important if we actually get serious about attempting to optimize this down


// Input format includes the following

// chunk_count_per_program:
//   represents the number of program chunks per program
//   1d array of uint8, dimension [PROGRAM_COUNT_PER_BLOCK]

// data_ops:
//   represents a sequence of +-<> ops
//   3d array of uint8, dimension [DATA_CELL_COUNT+1][MAX_CHUNK_COUNT_PER_PROGRAM][PROGRAM_COUNT_PER_BLOCK]
//     each program member array at index [program_index] of dimension [DATA_CELL_COUNT+1][MAX_CHUNK_COUNT_PER_PROGRAM] only has a meaningful size of chunk_count_per_program[program_index]
//     each program chunk member array at index [program_index][program_chunk_index] of dimension [DATA_CELL_COUNT+1] has the following format:
//       [ data_ptr_diff, data_cell_0_diff, data_cell_1_diff, ..., data_cell_{DATA_CELL_COUNT-1}_diff ]
//       the first element is data_ptr_diff, this could be moved into its own array, however my thought is that locality of data is relevent here, especially since modification is a simple add similar to data modification
//       that said there may not be caching concerns as this should probably be only relevent in a single thread, and the volume of data per program is relatively small, worth testing a version which separates this

// io_ops:
//   represents a . or , op
//   2d array of IO_OP_t, dimension [MAX_CHUNK_COUNT_PER_PROGRAM][PROGRAM_COUNT_PER_BLOCK]
//     each program member array at index [program_index] of dimension [MAX_CHUNK_COUNT_PER_PROGRAM] only has a meaningful size of chunk_count_per_program[program_index]
//     the members of this array each associated with a single program chunk represent a read from input (,) to the current data cell or a write to output (.) from the current data cell
//       in both cases, the respective io pointer will be incremented and wrap

// control_ops:
//   represents a [ or ] op
//   3d array of uint8, dimension [2][MAX_CHUNK_COUNT_PER_PROGRAM][PROGRAM_COUNT_PER_BLOCK]
//     each program member array at index [program_index] of dimension [2][MAX_CHUNK_COUNT_PER_PROGRAM] only has a meaningful size of chunk_count_per_program[program_index]
//     each program chunk member array at index [program_index][program_chunk_index] of dimension [2] represents the next chunk if the current data cell is 0, and the next chunk if the current data cell is nonzero
//       these members are chunk indexes within the current program and are guaranteed to be less than or equal to chunk_count_per_program[program_index]
//       if the next index is equla to chunk_count_per_program[program_index] the program terminates

// NOTE: the following could be made significantly more memory efficient if we wish to have the same input data for each block, or even for the entire block grid, though there may be data access speed implications, especially in the latter case

// input_data_count_per_program
//   represents the length of input_data per program
//   1d array of uint8, dimension [PROGRAM_COUNT_PER_BLOCK]

// input_data
//   represents the sequence to be read using ,
//   2d array of uint8, dimension [MAX_INPUT_DATA_COUNT][PROGRAM_COUNT_PER_BLOCK]
//     each program member array at index [program_index] of dimension [MAX_INPUT_DATA_COUNT] is 0 terminated and only has a meaningful size up to the first 0
//     the input_data_ptr will wrap to the beginning whenever it encounters a 0


// Execution notes

// The execution flow is as follows:
//   data_ops[current_chunk_index] is applied to the working data_cell_array
//   io_ops[current_chunk_index] is applied if present, copying between the current data cell, input stream, or output stream as necessary
//   control_ops[current_chunk_index] is used to find the next chunk to execute, if the next chunk is out of bounds or if the chunks_executed_counter reaches MAX_CHUNKS_EXECUTED_COUNTER the program is terminated

// it might be worth experimenting with uint16 or uint32 rather than uint8 for potential speedups regarding data alignment,
//   I do not know how important data alignment for integral operations is on the gpu as opposed to the cpu, however my intuition is that the gpu should support per byte operations decently well, and that the memory overhead would not justify these changes
//   in regards to this remember that memory overhead can translate to performance overhead very easily when factoring in per thread or per block processor caches as well as actual coprocessor data transfer

#include <cooperative_groups.h>
using namespace cooperative_groups;

// NOTE: We use preprocessing macros purely for performance reasons

#define PROGRAM_COUNT_PER_BLOCK 256
#define DATA_CELL_COUNT 31 /* NOTE: this plus 1 should be 4 byte aligned; might be particularly worth attempting 15, as cuda can handle up to 16 bytes in a single instruction */
#define MAX_CHUNK_COUNT_PER_PROGRAM 16 /* NOTE: we can increase this if necessary, but every programs' memory scales linearly with this no matter how much of it they use */
#define MAX_INPUT_DATA_COUNT 16
#define MAX_OUTPUT_DATA_COUNT 16
#define MAX_CHUNKS_EXECUTED_COUNTER 1024 /* NOTE: we can experiment a good deal with this value */

// TODO: enum is probably the same as this after compile and definitely better practice
#define IO_OP_t uint8_t
#define IO_OP_NONE 0
#define IO_OP_INPUT 1
#define IO_OP_OUTPUT 2

#define WORKING_DATA_PTR_OFFSET DATA_CELL_COUNT
#define DATA_CELL_MIN (DATA_CELL_COUNT / 2)
#define DATA_CELL_MAX (3 * DATA_CELL_COUNT / 2)
#define DATA_OP_OFFEST (DATA_CELL_COUNT / 2)

// TODO: current version has a lot of problems as discussed in various comments and the wall of text above, however I want to get a simple running version before prematurely addressing them
__global__ void ExecuteBfKernal(
	uint8_t chunk_count_per_program[PROGRAM_COUNT_PER_BLOCK],
	uint8_t data_ops[DATA_CELL_COUNT+1][MAX_CHUNK_COUNT_PER_PROGRAM][PROGRAM_COUNT_PER_BLOCK],
	IO_OP_t io_ops[MAX_CHUNK_COUNT_PER_PROGRAM][PROGRAM_COUNT_PER_BLOCK],
	uint8_t control_ops[2][MAX_CHUNK_COUNT_PER_PROGRAM][PROGRAM_COUNT_PER_BLOCK],
	uint8_t input_data_count_per_program[PROGRAM_COUNT_PER_BLOCK],
	uint8_t input_data[MAX_INPUT_DATA_COUNT][PROGRAM_COUNT_PER_BLOCK],
	uint8_t output_data[MAX_OUTPUT_DATA_COUNT][PROGRAM_COUNT_PER_BLOCK],
)
{
	uint8_t program_index = threadIdx.x;
	uint8_t current_chunk_ptr = 0;

	uint8_t current_working_data_ptr = WORKING_DATA_PTR_OFFSET;
	// Figure out if this needs to be explicitly zero initialized
	uint8_t current_working_data[DATA_CELL_COUNT * 2];
	
	uint8_t current_input_data_ptr = 0;
	uint8_t current_output_data_ptr = 0;

	// NOTE: see high level notes for my thoughts on the potential performance optimizations of this loop
	for (uint32_t i = MAX_CHUNKS_EXECUTED_COUNTER; i > 0; i--)
	{
		// execute data ops
		// NOTE: this might be worth explicitely splitting up over a warp, or some form of cooperative group, though this is certainly non-trivial
		for (uint32_t working_data_index = 1; i < DATA_CELL_COUNT; i++)
		{
			current_working_data[current_working_data_ptr + working_data_index - DATA_OP_OFFEST - 1] += data_ops[program_index][current_chunk_ptr][working_data_index];
		}
		current_working_data_ptr += data_ops[program_index][current_chunk_ptr][0];

		// verify data pointer is still in bounds
		// NOTE: this worries me perf-wise
		if (current_working_data_ptr < DATA_CELL_MIN || current_working_data_ptr >= DATA_CELL_MAX)
		{
			break;
		}

		// execute io ops
		// NOTE: cuda should be smart enough not to emit diverging branches here, but this should probably be confirmed
		IO_OP_t io_op = io_ops[program_index][current_chunk_ptr];
		if (io_op == IO_OP_INPUT)
		{
			current_working_data[current_working_data_ptr] = input_data[program_index][current_input_data_ptr];
			current_input_data_ptr = (current_input_data_ptr + 1) % input_data_count_per_program[program_index];
		}
		if (io_op == IO_OP_OUTPUT)
		{
			current_output_data[current_output_data_ptr] = current_working_data[current_working_data_ptr];
			current_output_data_ptr = (current_output_data_ptr + 1) % MAX_OUTPUT_DATA_COUNT;
		}

		// execute control flow ops
		// NOTE: even if cuda is smart enough not to emit diverging branches here, I think it's better to just access using a conditional directly
		current_chunk_ptr = control_ops[program_index][program_chunk_index][ current_working_data[current_working_data_ptr] != 0 ]
		// Check if program has regularly termintated
		// NOTE: this worries me perf-wise
		if (current_chunk_ptr >= chunk_count_per_program[program_index])
		{
			break;
		}
	}
}

struct ExecuteBfParams
{
	uint8_t chunk_count_per_program[PROGRAM_COUNT_PER_BLOCK],
	uint8_t data_ops[DATA_CELL_COUNT+1][MAX_CHUNK_COUNT_PER_PROGRAM][PROGRAM_COUNT_PER_BLOCK],
	IO_OP_t io_ops[MAX_CHUNK_COUNT_PER_PROGRAM][PROGRAM_COUNT_PER_BLOCK],
	uint8_t control_ops[2][MAX_CHUNK_COUNT_PER_PROGRAM][PROGRAM_COUNT_PER_BLOCK],
	uint8_t input_data_count_per_program[PROGRAM_COUNT_PER_BLOCK],
	uint8_t input_data[MAX_INPUT_DATA_COUNT][PROGRAM_COUNT_PER_BLOCK],
	uint8_t output_data[MAX_OUTPUT_DATA_COUNT][PROGRAM_COUNT_PER_BLOCK],
}

void ExecuteBfCuda(ExecuteBfCudaParams & params)
{
	// Essentially just moves data from host to device and back after execution

	uint8_t* DEVICE_chunk_count_per_program;
	uint8_t* DEVICE_data_ops;
	IO_OP_t* DEVICE_io_ops;
	uint8_t* DEVICE_control_ops;
	uint8_t* DEVICE_input_data_count_per_program;
	uint8_t* DEVICE_input_data;
	uint8_t* DEVICE_output_data;

	cudaMalloc(&DEVICE_chunk_count_per_program, PROGRAM_COUNT_PER_BLOCK);
	cudaMalloc(&DEVICE_data_ops, (DATA_CELL_COUNT+1) * MAX_CHUNK_COUNT_PER_PROGRAM * PROGRAM_COUNT_PER_BLOCK);
	cudaMalloc(&DEVICE_io_ops, MAX_CHUNK_COUNT_PER_PROGRAM * PROGRAM_COUNT_PER_BLOCK);
	cudaMalloc(&DEVICE_control_ops, 2 * MAX_CHUNK_COUNT_PER_PROGRAM * PROGRAM_COUNT_PER_BLOCK);
	cudaMalloc(&DEVICE_input_data_count_per_program, PROGRAM_COUNT_PER_BLOCK);
	cudaMalloc(&DEVICE_input_data, MAX_INPUT_DATA_COUNT * PROGRAM_COUNT_PER_BLOCK);
	cudaMalloc(&DEVICE_output_data, MAX_OUTPUT_DATA_COUNT * PROGRAM_COUNT_PER_BLOCK);

	cudaMemcpy(DEVICE_chunk_count_per_program, params.chunk_count_per_program, PROGRAM_COUNT_PER_BLOCK, cudaMemcpyHostToDevice);
	cudaMemcpy(DEVICE_data_ops, params.data_ops, (DATA_CELL_COUNT+1) * MAX_CHUNK_COUNT_PER_PROGRAM * PROGRAM_COUNT_PER_BLOCK, cudaMemcpyHostToDevice);
	cudaMemcpy(DEVICE_io_ops, params.io_ops, MAX_CHUNK_COUNT_PER_PROGRAM * PROGRAM_COUNT_PER_BLOCK, cudaMemcpyHostToDevice);
	cudaMemcpy(DEVICE_control_ops, params.control_ops, 2 * MAX_CHUNK_COUNT_PER_PROGRAM * PROGRAM_COUNT_PER_BLOCK, cudaMemcpyHostToDevice);
	cudaMemcpy(DEVICE_input_data_count_per_program, params.input_data_count_per_program, PROGRAM_COUNT_PER_BLOCK, cudaMemcpyHostToDevice);
	cudaMemcpy(DEVICE_input_data, params.input_data, MAX_INPUT_DATA_COUNT * PROGRAM_COUNT_PER_BLOCK, cudaMemcpyHostToDevice);

	cudaMemset(DEVICE_output_data, 0, MAX_OUTPUT_DATA_COUNT * PROGRAM_COUNT_PER_BLOCK);

	ExecuteBf<<<1, PROGRAM_COUNT_PER_BLOCK>>>(
		DEVICE_chunk_count_per_program,
		DEVICE_data_ops,
		DEVICE_io_ops,
		DEVICE_control_ops,
		DEVICE_input_data_count_per_program,
		DEVICE_input_data,
		DEVICE_output_data,
	);

	cudaMemcpy(params.output_data, DEVICE_output_data, MAX_OUTPUT_DATA_COUNT * PROGRAM_COUNT_PER_BLOCK, cudaMemcpyDeviceToHost);

	cudaFree(DEVICE_chunk_count_per_program);
	cudaFree(DEVICE_data_ops);
	cudaFree(DEVICE_io_ops);
	cudaFree(DEVICE_control_ops);
	cudaFree(DEVICE_input_data_count_per_program);
	cudaFree(DEVICE_input_data);
	cudaFree(DEVICE_output_data);
}

void BfSourceToExecuteParams(char const * const source, ExecuteBfParams& params, uint32_t program_index)
{
	char source_char;
	uint32_t source_index = 0;
	
	uint8_t current_chunk_index = 0;
	uint8_t current_chunk_data_op[DATA_CELL_COUNT];
	memset(current_chunk_data_op, 0, DATA_CELL_COUNT);
	uint8_t current_chunk_data_ptr = DATA_OP_OFFEST;

	while (source_char = source[source_index++])
	{
		switch (source_char)
		{
			case '+':
				current_chunk_data_op[current_chunk_data_ptr] += 1;
				break;

			case '-':
				current_chunk_data_op[current_chunk_data_ptr] -= 1;
				break;

			case '>':
				current_chunk_data_ptr += 1;
				if (current_chunk_data_ptr == DATA_CELL_COUNT - 1) 
				{
					// TODO: fail gracefully
					throw 1;
				}
				break;

			case '<':
				if (current_chunk_data_ptr == 0)
				{
					// TODO: fail gracefully
					throw 1;
				}
				current_chunk_data_ptr -= 1;
				break;

			case '.':
				// commit data op
				uint8_t* param_dest = &(params.data_ops[program_index][current_chunk_index][1])
				memcpy(current_chunk_data_op, param_dest, DATA_CELL_COUNT);
				params.data_ops[program_index][current_chunk_index][0] = current_chunk_data_ptr;

				// clear data op
				memset(current_chunk_data_op, 0, DATA_CELL_COUNT);
				current_chunk_data_ptr = DATA_OP_OFFEST;

				
		}
	}
}

// This should not be used for actual execution as it loads only a single program onto the gpu, it's only purpose is a quick test functionality
void ExecuteBfSingle(char const * const program)
{

}

int main()
{

}
