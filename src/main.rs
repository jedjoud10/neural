use std::{io::Cursor, path::Path};
use rand::seq::SliceRandom;
use image::{io::Reader as ImageReader, DynamicImage, ImageBuffer, Rgb, RgbImage};
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
                    let img = img.resize_exact(64, 64, image::imageops::FilterType::Gaussian);
                    //let img = img.grayscale();
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
    

    let mut nn = Network::input(64*64*3)
        .dense(8)
        .output(4);

    let mut rng = rand::thread_rng();
    let mut datas = rx.try_iter().collect::<Vec<_>>();
    datas.shuffle(&mut rng);
    nn.find_lowest_cost_nn(&datas, 0.002, 100, 0.06);
    nn.test_all(&datas);    

    let layer = &nn.0[0];
    for (i, n) in layer.0.iter().enumerate() {
        let mut image = RgbImage::new(64, 64);
        
        for p in 0..(64*64) {
            let x = p % 64;
            let y = p / 64;
            let r = n.weights[p*3] * 255.0;
            let g = n.weights[p*3+1] * 255.0;
            let b = n.weights[p*3+2] * 255.0;
            image.put_pixel(x as u32, y as u32, Rgb([r as u8,g as u8,b as u8]));

        }

        let image = DynamicImage::from(image).resize(64, 64, image::imageops::FilterType::Nearest);
        image.save(format!("./src/a{i}.bmp")).unwrap();
    }
}
