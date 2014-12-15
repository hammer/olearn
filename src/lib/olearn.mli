type sample = { x : float; y : float; }
type model = { theta : float; beta : float; }
type hyperparameters = { epochs : int; learning_rate : float; }
type epoch_state = { m : model; h : hyperparameters; }
type fits = epoch_state list

val shuffle : 'a array -> 'a array
val r2_score : float array -> float array -> float
val predict : model -> float -> float
val step_parameters : epoch_state -> sample -> epoch_state
val single_pass : epoch_state list -> sample array -> epoch_state list
val all_passes : int -> epoch_state list -> sample array -> epoch_state list
val fit_regressor : sample array -> hyperparameters -> epoch_state list
