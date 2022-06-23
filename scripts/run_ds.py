import sys
sys.path.insert(0, "../DeepSequence")

import model
import helper

args = sys.argv

if len(args) < 5:
    raise ValueError('Syntax is python run_ds.py aln_file model_prefix mutation_file out_file')

alignment_file = args[1]
file_prefix = args[2]
mutation_file = args[3]
out_file = args[4]

data_helper = helper.DataHelper(
                alignment_file=alignment_file,
                working_dir=".",
                calc_weights=False
                )

model_params = {
        "batch_size"        :   100,
        "encode_dim_zero"   :   1500,
        "encode_dim_one"    :   1500,
        "decode_dim_zero"   :   100,
        "decode_dim_one"    :   500,
        "n_patterns"        :   4,
        "n_latent"          :   30,
        "logit_p"           :   0.001,
        "sparsity"          :   "logit",
        "encode_nonlin"     :   "relu",
        "decode_nonlin"     :   "relu",
        "final_decode_nonlin":  "sigmoid",
        "output_bias"       :   True,
        "final_pwm_scale"   :   True,
        "conv_pat"          :   True,
        "d_c_size"          :   40
        }

ds_model = model.VariationalAutoencoder(data_helper,
    batch_size              =   model_params["batch_size"],
    encoder_architecture    =   [model_params["encode_dim_zero"],
                                model_params["encode_dim_one"]],
    decoder_architecture    =   [model_params["decode_dim_zero"],
                                model_params["decode_dim_one"]],
    n_latent                =   model_params["n_latent"],
    n_patterns              =   model_params["n_patterns"],
    convolve_patterns       =   model_params["conv_pat"],
    conv_decoder_size       =   model_params["d_c_size"],
    logit_p                 =   model_params["logit_p"],
    sparsity                =   model_params["sparsity"],
    encode_nonlinearity_type       =   model_params["encode_nonlin"],
    decode_nonlinearity_type       =   model_params["decode_nonlin"],
    final_decode_nonlinearity      =   model_params["final_decode_nonlin"],
    output_bias             =   model_params["output_bias"],
    final_pwm_scale         =   model_params["final_pwm_scale"],
    working_dir             =   ".")

print ("Model built")

print ("Loading model parameters.")
ds_model.load_parameters(file_prefix=file_prefix)
print ("Parameters loaded.")

print ("Making predictions...")
mutant_names, delta_elbos = data_helper.custom_mutant_matrix(mutation_file, ds_model, N_pred_iterations=500)

print ("Writing output...")
with open(out_file, 'w') as f:
	for mut, score in zip(mutant_names, delta_elbos):
		f.write(mut + '\t' + str(score) + '\n')

