use rand::Rng;

fn activation(val: f32) -> f32 {
    //val.max(0.0)
    val.clamp(0.0, 1.0)
}

fn main() {
    let mut rng = rand::thread_rng();
    let mut inputs = Vec::<f32>::new();
    inputs.push(0.0);
    inputs.push(0.0);
    inputs.push(0.0);

    inputs.push(0.0);
    inputs.push(1.0);
    inputs.push(0.0);
    
    inputs.push(1.0);
    inputs.push(0.0);
    inputs.push(0.0);

    inputs.push(1.0);
    inputs.push(1.0);
    inputs.push(1.0);
    
    let mut outputs = Vec::<f32>::new();

    outputs.push(0.0);
    outputs.push(0.0);
    outputs.push(0.0);
    outputs.push(1.0);
    

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
    
}
