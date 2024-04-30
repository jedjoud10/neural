use std::{io::Cursor, path::Path};
use image::io::Reader as ImageReader;
mod network;
pub use network::*;

fn main() {
    
    let mut rng = rand::thread_rng();
    let mut datas = Vec::<Data>::new();
    
    for k in 0..4 {
        let color = ["blue", "green", "red", "yellow"][k];
        for i in 1..105 {
            let img = ImageReader::open(format!("./src/images/{color}{i}.jpg")).unwrap().decode().unwrap();
            let img = img.resize_exact(8, 8, image::imageops::FilterType::Gaussian);
            let img = img.into_rgb32f();
            let input = img.to_vec();
            dbg!(input.len());
            let mut output = vec![0.0f32; 4];
            output[k] = 1.0;

            datas.push(Data {
                inputs: input,
                outputs: output,
            });
        }
    }

    let mut nn = Network::input(8*8*3)
        .dense(32)
        .output(4);

    nn.find_lowest_cost_nn(&datas, 10000, 10, 0.02, &mut rng);
    nn.test_all(&datas);

    /*

    // how many input -> output lines
    let data_count = 4;
    let input_shape = 3;

    // layers:
    // input: 2 neurons
    // output: 1
    for _ in 0..10 {
        
        for _ in 0..100 {
            let mut bias = vec![0.0; 3];
            let mut factor = vec![0.0; 3];
    
            for i in 0..input_shape {
                bias[i] = rng.gen_range(-1.0..1.0);
                factor[i] = rng.gen_range(-1.0..1.0);
            }
    
            let mut total = 0.0f32;
            for i in 0..4 {
                let input = &inputs[(input_shape * i)..(input_shape * (i+1))];
                let output = outputs[i];
    
                let sum = (0..3).into_iter()
                    .map(|i| activation(bias[i] + factor[i] * input[i]))
                    .sum::<f32>();
    
                total += (output - sum).abs();
            }    
            dbg!(total);
    
            if total <= 1.0 {
                dbg!(bias);
                dbg!(factor);
            }
        }
    }
    */
    
}
