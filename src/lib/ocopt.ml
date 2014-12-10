type sample = {
  x : float;
  y : float;
}

type model = {
  theta : float;
  beta  : float;
}

type hyperparameters = {
  learning_rate : float;
}

(* Thanks, batteries (and Fisher-Yates) *)
let shuffle a =
  for n = Array.length a - 1 downto 1 do
    let k    = Random.int ( n + 1 ) in
    if k <> n then
      let buf  = Array.get a n in
      Array.set a n (Array.get a k);
      Array.set a k buf
  done;
  a

let print_samples ss =
  Array.iter (fun s -> Printf.printf "x: %f y: %f\n" s.x s.y) ss

let print_models ms =
  List.iter (fun m -> Printf.printf "theta: %f beta: %f\n" m.theta m.beta) ms

(* Parameter update *)
let score m x =
  m.theta *. x +. m.beta

let step_parameters m s h =
  let y_hat = score m s.x in
  { theta = m.theta -. h.learning_rate *. (y_hat -. s.y) *. s.x;
    beta = m.beta -. h.learning_rate *. (y_hat -. s.y)
  }

let single_pass m_0 samples h =
  let f ms s = (step_parameters (List.hd ms) s h) :: ms in
  Array.fold_left f m_0 samples
