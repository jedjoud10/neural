use std::{io::Cursor, path::Path};
use rand::seq::SliceRandom;
use image::io::Reader as ImageReader;
mod network;
pub use network::*;

fn main() {
    let (mut tx, mut rx) = std::sync::mpsc::channel::<Data>();
    
    rayon::scope(|s| {
        for k in 0..4 {
            let color = ["blue", "green", "red", "yellow"][k];
            for i in 1..105 {
                let tx = tx.clone();
                s.spawn(move |_| {
                    let path = format!("./src/images/{color}{i}.jpg");
                    println!("Loading image {path}");
                    let img = ImageReader::open(path).unwrap().decode().unwrap();
                    let img = img.resize_exact(8, 8, image::imageops::FilterType::Gaussian);
                    let img = img.into_rgb32f();
                    let input = img.to_vec();
                    let mut output = vec![0.0f32; 4];
                    output[k] = 1.0;
        
                    tx.send(Data {
                        inputs: input,
                        outputs: output,
                    }).unwrap();
                });
                
            }
        }
    });
    

    let mut nn = Network::input(8*8*3)
        .dense(8)
        .output(4);

    let mut rng = rand::thread_rng();
    let mut datas = rx.try_iter().collect::<Vec<_>>();
    datas.shuffle(&mut rng);
    nn.find_lowest_cost_nn(&datas, 0.1, 10, 0.02);
    nn.test_all(&datas);    
}
