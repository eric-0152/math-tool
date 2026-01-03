use crate::{matrix::Matrix, vector::Vector};
use rand::Rng;
use rand_distr::{Distribution, Normal};

pub struct MCMC {
    pub init_param: Vector,
    pub param_evo: Matrix,
    pub current_fx: f64,
}

pub fn get_guassian_kernel(mean: f64, std: f64) -> Result<Normal<f64>, String> {
    match Normal::new(mean, std) {
        Err(error_msg) => Err(error_msg.to_string()),
        Ok(normal) => Ok(normal),
    }
}

impl MCMC {
    pub fn init(
        init_param: Vector,
    ) -> MCMC {
        MCMC {
            init_param: init_param.clone(),
            param_evo: init_param.transpose(),
            current_fx: 0.0,
        }
    }
    
    pub fn guassian_move(self: &mut Self, guassian_kernel: &Normal<f64>, evaluate_func: &dyn Fn(&Vector) -> f64) -> (Vector, f64) {
        let mut new_param: Vector = self.param_evo.get_row_vector(self.param_evo.row() - 1).unwrap();
    
        for p in 0..new_param.size() {
            new_param.entries[p] += guassian_kernel.sample(&mut rand::rng());
        }
    
        let mut new_fx: f64 = (evaluate_func)(&new_param);
        if new_fx < self.current_fx {
            let mut generator: rand::prelude::ThreadRng = rand::rng();
            if (new_fx / self.current_fx) <= generator.random_range(0.0..=1.0) {
                new_param = self.param_evo.get_row_vector(self.param_evo.row() - 1).unwrap();
                new_fx = self.current_fx;
            }
        }
    
        (new_param, new_fx)
    }
}



pub fn burn_in(param_evo: &Matrix, burn_in_step: u32) -> Result<Matrix, String> {
    if burn_in_step > param_evo.row() as u32 {
        return Err("Input Error: Parameter burn_in_step should less than iteration".to_string());
    }

    let mut result_param_evo: Matrix = param_evo.clone();
    for _ in 0..burn_in_step {
        result_param_evo = result_param_evo.remove_row(0).unwrap();
    }

    Ok(result_param_evo)
}
