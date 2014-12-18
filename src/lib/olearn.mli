type input = float array
type inputs = input array
type output = float
type outputs = output array
type predicted_output = output
type predicted_outputs = predicted_output array
type num_epochs = int
type r2_score = float

type sample = { x : input; y : output; }
type model = { theta : float array; beta : float; }
type hyperparameters = { epochs : num_epochs; learning_rate : float; }
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
