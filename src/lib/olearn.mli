open Lacaml.D

type input = Lacaml_float64.vec
type inputs = Lacaml_float64.mat
type output = float
type outputs = Lacaml_float64.vec
type predicted_output = output
type predicted_outputs = Lacaml_float64.vec
type num_epochs = int
type r2_score = float
type eta_schedule = Constant | Inverse_scaling
type regularization = No_regularization | L2

type sample = { x : input; y : output; }
type model = { theta : Lacaml_float64.vec; }
type hyperparameters = {
    epochs : num_epochs;
    eta : float;
    eta_schedule : eta_schedule;
    power_t : float;
    lambda : float;
    regularization : regularization;
    time_steps : int;
  }
type epoch_state = { m : model; h : hyperparameters; }
type samples = sample array
type fits = epoch_state list

val shuffle : 'a array -> 'a array
val r2_score : outputs -> predicted_outputs -> r2_score
val predict : model -> input -> predicted_output
val fit_sample : epoch_state -> sample -> epoch_state
val fit_epoch : fits -> samples -> fits
val fit_epochs : num_epochs -> fits -> samples -> fits
val fit_regressor : epoch_state -> samples -> fits
