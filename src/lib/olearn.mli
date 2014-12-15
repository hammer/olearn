type sample = { x : float; y : float; }
type model = { theta : float; beta : float; }
type hyperparameters = { epochs : int; learning_rate : float; }
type epoch_state = { m : model; h : hyperparameters; }
type fits = epoch_state list

val shuffle : 'a array -> 'a array
val r2_score : float array -> float array -> float
val predict : model -> float -> float
val step_parameters : epoch_state -> sample -> epoch_state
val single_pass : fits -> sample array -> fits
val all_passes : int -> fits -> sample array -> fits
val fit_regressor : sample array -> hyperparameters -> fits
