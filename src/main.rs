use rand::Rng;

fn activation(val: f32) -> f32 {
    //val.max(0.0)
    val.clamp(0.0, 1.0)
}



/*
1 input node
1 output node
activation fn: s(x)

real truth: r
evaluated truth: k
cost => c = (r - k)^2
c' = 2(r - k)


n inputs
n outputs

*/

struct Data<'a> {
    inputs: &'a [f32],
    outputs: &'a [f32],
}

#[derive(Default, Clone)]
struct Neuron {
    // for dense layers this should be equal to the number of neurons of the last layer
    weights: Vec<f32>,
    bias: f32,
}

struct Layer(Vec<Neuron>);
struct Network(Vec<Layer>);
impl Network {
    fn input(neurons: usize) -> Self {
        let layer = Layer(vec![Neuron::default(); neurons]);
        Self(vec![layer])
    }

    fn dense(mut self, neurons: usize) -> Self {
        self
    }

    fn output(mut self, neurons: usize) -> Self {
        self
    }

    fn eval(&self, data: &Data) -> Vec<f32> {
        let mut last_layer_activations = data.inputs.to_vec();
        let mut temp_next = Vec::<f32>::new();
        
        for layer in self.0.iter() {
            for neuron in layer.0.iter() {
                
            }
        }

        last_layer_activations
    }
}

fn main() {
    let mut rng = rand::thread_rng();
    let mut datas = Vec::<Data>::new();
    datas.push(Data {
        inputs: &[0.0, 1.0],
        outputs: &[1.0],
    });

    datas.push(Data {
        inputs: &[1.0, 0.0],
        outputs: &[1.0],
    });

    datas.push(Data {
        inputs: &[0.0, 0.0],
        outputs: &[0.0],
    });
    
    let nn = Network::input(2)
        .dense(12)
        .output(1);
    let output = nn.eval(&datas[0]);
    dbg!(output);

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
