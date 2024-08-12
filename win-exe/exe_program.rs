use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_event, cl_ulong, CL_NON_BLOCKING};
use opencl3::Result;
use std::os::raw::c_void;
use std::ptr;

const TOTAL_WORK_TO_DO: usize = 1_000_000_000;
const TESTS: usize = 10;

const PROGRAM_SOURCE: &str = r#"
    ulong gen_random_ulong(ulong2 seed, ulong last_roll) {
        ulong seed_x = seed.x + last_roll;
        ulong t = seed_x ^ (seed_x << 11);
        ulong result = seed.y ^ (seed.y >> 19) ^ (t ^ (t >> 8));

        return result;
    }

    kernel void roll_dice(global ulong* output, global ulong2 const* seeds, global ulong const* work_to_do) {
        const size_t i = get_global_id(0);

        ulong2 seed = seeds[i];

        uchar rolls[64];

        ulong last_roll = 0;

        uchar max_seen = 0;

        ulong num_tries = work_to_do[i];

        for (int j = 0; j < num_tries; j++) {
            uchar ones = 0;

            for (int k = 0; k < 8; k++) {
                ulong result = gen_random_ulong(seed, last_roll);

                last_roll = result;

                ulong* rolls_ptr = (ulong*)rolls;
                rolls_ptr[k] = result;
            }


            for (int k = 0; k < 57; k++) {
                ones += (rolls[k] & 0b00000011)==0;
                ones += (rolls[k] & 0b00001100)==0;
                ones += (rolls[k] & 0b00110000)==0;
                ones += (rolls[k] & 0b11000000)==0;
            }

            ulong last_roll = rolls[57];
            ones += (last_roll & 0b00000011) == 0;
            ones += (last_roll & 0b00001100) == 0;
            ones += (last_roll & 0b00110000) == 0;

            if (ones > max_seen) {
                max_seen = ones;
            }
        }

        output[i] = max_seen;
    }

"#;

const KERNEL_NAME: &str = "roll_dice";

fn main() -> Result<()> {

    let time = std::time::Instant::now();

    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");

    let device = Device::new(device_id);

    let compute_units  = device.max_compute_units()?;

    let max_work_group_size = device.max_work_group_size()?;

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");


    let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
        .expect("CommandQueue::create_default failed");


    // Build the OpenCL program source and create the kernel.
    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "")
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");
    
    let work_vec: Vec<usize> = {        
        let mut output = vec![];

        let total_workers = compute_units as usize * max_work_group_size as usize;

        let work_per_worker = TOTAL_WORK_TO_DO / total_workers as usize;

        for _ in 0..total_workers {
            output.push(work_per_worker);
        }

        let remainder = TOTAL_WORK_TO_DO % total_workers;

        for i in 0..remainder {
            output[i] += 1;
        }

        output
    };    


    println!("running on: {:?}", device.name()?);

    println!("finished building OpenCL program, running tests");

    let elapsed = time.elapsed();
    println!("building took {}ms", elapsed.as_millis());

    let mut max_nums = vec![];

    let mut average_time = 0;

    for i in 0..TESTS {

        let time = std::time::Instant::now();

        let seeds = gen_seeds(work_vec.len());

        let seeds_buffer = unsafe {
            Buffer::<[cl_ulong; 2]>::create(
                &context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, seeds.len(), seeds.as_ptr() as *mut c_void,
            )?
        };

        
        let output_buffer = unsafe {
            Buffer::<cl_ulong>::create(
                &context, CL_MEM_WRITE_ONLY, work_vec.len() as usize, ptr::null_mut(),
            )?
        };


        let work_to_do_buffer = unsafe {
            Buffer::<cl_ulong>::create(
                &context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, work_vec.len(), work_vec.as_ptr() as *mut c_void,
            )?
        };

        // set the kernel arguments
        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&output_buffer)
                .set_arg(&seeds_buffer)
                .set_arg(&work_to_do_buffer)
                .set_global_work_size(work_vec.len())
                .enqueue_nd_range(&queue)?
        };

        let mut events: Vec<cl_event> = Vec::default();
        events.push(kernel_event.get());

        let mut output = vec![0; work_vec.len()];

        unsafe {
            queue.enqueue_read_buffer(&output_buffer, CL_NON_BLOCKING, 0, &mut output, &events)?;    
        }

        


        let max_output = output.iter().max().unwrap();
        let sum_work = work_vec.iter().sum::<usize>();

        max_nums.push(*max_output);


        let elapsed = time.elapsed();

        println!("test {} took {}ms", i, elapsed.as_millis());


        average_time += elapsed.as_millis();

    }

    println!("average time: {}ms", average_time / TESTS as u128);

    println!("max nums: {:?}", max_nums);


    
    Ok(())
}

fn gen_seeds(num_workers: usize) -> Vec<[cl_ulong; 2]> {
    let mut seeds = Vec::with_capacity(num_workers);
    for _ in 0..num_workers {
        let seed_x = rand::random::<u64>();
        let seed_y = rand::random::<u64>();

        seeds.push([seed_x, seed_y]);
    }
    seeds
}