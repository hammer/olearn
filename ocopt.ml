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
  Printf.printf "Starting iteration with samples:\n";
  print_samples samples;
  Printf.printf "And model history:\n";
  print_models m_0;
  Array.fold_left (fun ms s -> (step_parameters (List.hd ms) s h) :: ms) m_0 samples

(* Initial conditions *)
let initial_model =
  { theta = 0.6; beta = 25. }

let initial_hyperparameters =
  { learning_rate = 0.0001 }

(* Simple regression example from http://stattrek.com/regression/regression-example.aspx *)
let () =
  let x = BatArray.enum [|95.; 85.; 80.; 70.; 60.|] in
  let y = BatArray.enum [|85.; 95.; 70.; 65.; 70.|] in
  let make_sample (x, y) = { x = x; y = y } in
  let samples = BatArray.of_enum (BatEnum.map make_sample (BatEnum.combine (x, y))) in
  let m_0 = initial_model in
  let h_0 = initial_hyperparameters in
  let shuffled_samples = shuffle samples in
  let iter_1 = single_pass [m_0] shuffled_samples h_0 in
  let iter_2 = single_pass iter_1 shuffled_samples h_0 in
  Printf.printf "Model history:\n";
  print_models iter_2
