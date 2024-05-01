use rand::{rngs::ThreadRng, seq::SliceRandom, Rng};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

fn activation(val: f32) -> f32 {
    //val.max(0.0)
    //val.clamp(0.0, 1.0)
    1.0f32 / (1.0f32 + std::f32::consts::E.powf(-val))
}

pub struct Data {
    pub inputs: Vec<f32>,
    pub outputs: Vec<f32>,
}

#[derive(Default, Clone)]
pub struct Neuron {
    // for dense layers this should be equal to the number of neurons of the last layer
    weights: Vec<f32>,
    bias: f32,
}

#[derive(Clone)]
pub struct Layer(Vec<Neuron>);

#[derive(Clone)]
pub struct Network(Vec<Layer>, usize, usize);
impl Network {
    pub fn input(neurons: usize) -> Self {
        Self(Vec::new(), neurons, 0)
    }

    pub fn dense(mut self, neurons: usize) -> Self {
        let last_layer_neurons = self.0.last().map(|x| x.0.len()).unwrap_or(self.1);
        self.0.push(Layer(vec![Neuron { weights: vec![0.0; last_layer_neurons], bias: 0.0 }; neurons]));
        self
    }

    pub fn output(mut self, neurons: usize) -> Self {
        let last_layer_neurons = self.0.last().unwrap().0.len();
        self.0.push(Layer(vec![Neuron { weights: vec![0.0; last_layer_neurons], bias: 0.0 }; neurons]));
        self.2 = neurons;
        self
    }

    pub fn randomize(&mut self, rng: &mut ThreadRng, spread: f32) {
        for layer in self.0.iter_mut() {
            for neuron in layer.0.iter_mut() {
                neuron.bias += rng.gen_range(-spread..=spread);
                neuron.weights.iter_mut().for_each(|x| *x += rng.gen_range(-spread..=spread));
            }
        }
    }

    pub fn save(&self) -> Vec<f32> {
        let mut data = Vec::<f32>::new();
        
        for layer in self.0.iter() {
            for neuron in layer.0.iter() {
                data.extend(&neuron.weights);
                data.push(neuron.bias);
            }
        }

        data
    }

    pub fn load(&mut self, data: Vec<f32>) {
        let mut last_index = 0;

        for layer in self.0.iter_mut() {
            for neuron in layer.0.iter_mut() {
                let length = neuron.weights.len();
                neuron.weights.copy_from_slice(&data[last_index..(last_index + length)]);
                neuron.bias = data[last_index + length];
                last_index += length+1;
            }
        }
    }

    pub fn eval(&self, data: &Data) -> Vec<f32> {
        let mut last_layer_activations = data.inputs.to_vec();
        assert_eq!(last_layer_activations.len(), data.inputs.len());
        
        for (i, layer) in self.0.iter().enumerate() {
            let temp_next = layer.0.par_iter().map(|neuron| {
                neuron.bias + neuron.weights.par_iter().enumerate().map(|(i, &weight)| last_layer_activations[i] * weight).sum::<f32>()
            });

            let temp_next = temp_next.map(activation).collect::<Vec<_>>();
            last_layer_activations = temp_next;
        }

        last_layer_activations
    }

    pub fn cost(&self, data: &Data) -> f32 {
        let predicted = self.eval(data);
        let real = &data.outputs;

        predicted.iter().zip(real).map(|(a, b)| (a-b).powf(2.0)).sum::<f32>()
    }

    pub fn cost_all(&self, datas: &[Data]) -> f32 {
        let datas = &datas[0..20];
        return datas.par_iter().map(|x| self.cost(x)).sum::<f32>() / (datas.len() as f32);
    }

    pub fn test_all(&self, datas: &[Data]) {
        for data in datas {
            let predicted = self.eval(data).iter().enumerate().max_by(|a, b| (a.1).total_cmp(b.1)).unwrap().0;
            let real = &data.outputs.iter().enumerate().max_by(|a, b| (a.1).total_cmp(b.1)).unwrap().0;
            
            println!("Real: {real}, predict: {predicted}. {}", *real == predicted);
        }
    }

    pub fn find_lowest_cost_nn(&mut self, datas: &[Data], threshold: f32, entities: usize, randomized: f32) {
        let mut saved = self.save();
        let mut min = f32::MAX;
        let mut diff = 1.0;
        let mut rng = rand::thread_rng();
        while min > threshold {
            for k in 0..entities {
                let mut nn = self.clone();
                //let a = saved.iter().zip(delta.iter()).map(|(a, b)| *a + b * -0.2).collect::<Vec<_>>();
                nn.load(saved.clone());
                nn.randomize(&mut rng, randomized * (diff + 0.1));
                
                let score = nn.cost_all(&datas);
                //println!("Score: {score}");
            
                if score < min {
                    let temp = nn.save(); 
                    saved = temp;
                    min = score;
                    //diff = (min - threshold).abs() / ((min + threshold) / 2.0);
                    diff = (min - threshold).abs() * 10.0;
                    println!("Best Score: {min}");
                }
            }
        }
        
        
        self.load(saved);
    }
}