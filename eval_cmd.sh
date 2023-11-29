#!/bin/bash
echo "Agent Name: $1"
ENV1="PHlab_attitude_nominal"
ENV2="PHlab_attitude_ice"
ENV3="PHlab_attitude_cg-shift"
ENV4="PHlab_attitude_sa"
ENV5="PHlab_attitude_se"
ENV6="PHlab_attitude_be"
ENV7="PHlab_attitude_jr"
ENV8="PHlab_attitude_high-q"
ENV9="PHlab_attitude_low-q"
ENV10="PHlab_attitude_noise"
ENV11="PHlab_attitude_gust"
ENV12="PHlab_attitude_cg-for"
ENV13="PHlab_attitude_cg"



# python3 evaluate_es.py --agent_name=CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_100iters_deep2h64_new_covSaving_pop50_parents10_seed42  --env_name=PHlab_attitude_nominal --save_plots --use_best_mu --save_trajectory --save_stats --plot_spectra

python3 evaluate_es.py --agent_name=$1 --env_name=$ENV1 --save_plots --use_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV1 --save_plots --use_best_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV1 --save_plots --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV1 --save_plots --use_best_elite --save_trajectory --save_stats --plot_spectra




python3 evaluate_es.py --agent_name=$1 --env_name=$ENV2 --save_plots --use_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV2 --save_plots --use_best_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV2 --save_plots --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV2 --save_plots --use_best_elite --save_trajectory --save_stats --plot_spectra


python3 evaluate_es.py --agent_name=$1 --env_name=$ENV3 --save_plots --use_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV3 --save_plots --use_best_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV3 --save_plots --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV3 --save_plots --use_best_elite --save_trajectory --save_stats --plot_spectra


python3 evaluate_es.py --agent_name=$1 --env_name=$ENV4 --save_plots --use_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV4 --save_plots --use_best_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV4 --save_plots --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV4 --save_plots --use_best_elite --save_trajectory --save_stats --plot_spectra


python3 evaluate_es.py --agent_name=$1 --env_name=$ENV5 --save_plots --use_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV5 --save_plots --use_best_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV5 --save_plots --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV5 --save_plots --use_best_elite --save_trajectory --save_stats --plot_spectra


python3 evaluate_es.py --agent_name=$1 --env_name=$ENV6 --save_plots --use_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV6 --save_plots --use_best_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV6 --save_plots --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV6 --save_plots --use_best_elite --save_trajectory --save_stats --plot_spectra


python3 evaluate_es.py --agent_name=$1 --env_name=$ENV7 --save_plots --use_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV7 --save_plots --use_best_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV7 --save_plots --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV7 --save_plots --use_best_elite --save_trajectory --save_stats --plot_spectra


python3 evaluate_es.py --agent_name=$1 --env_name=$ENV8 --save_plots --use_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV8 --save_plots --use_best_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV8 --save_plots --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV8 --save_plots --use_best_elite --save_trajectory --save_stats --plot_spectra


python3 evaluate_es.py --agent_name=$1 --env_name=$ENV9 --save_plots --use_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV9 --save_plots --use_best_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV9 --save_plots --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV9 --save_plots --use_best_elite --save_trajectory --save_stats --plot_spectra


python3 evaluate_es.py --agent_name=$1 --env_name=$ENV10 --save_plots --use_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV10 --save_plots --use_best_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV10 --save_plots --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV10 --save_plots --use_best_elite --save_trajectory --save_stats --plot_spectra


python3 evaluate_es.py --agent_name=$1 --env_name=$ENV11 --save_plots --use_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV11 --save_plots --use_best_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV11 --save_plots --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV11 --save_plots --use_best_elite --save_trajectory --save_stats --plot_spectra


python3 evaluate_es.py --agent_name=$1 --env_name=$ENV12 --save_plots --use_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV12 --save_plots --use_best_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV12 --save_plots --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV12 --save_plots --use_best_elite --save_trajectory --save_stats --plot_spectra


python3 evaluate_es.py --agent_name=$1 --env_name=$ENV13 --save_plots --use_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV13 --save_plots --use_best_mu --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV13 --save_plots --save_trajectory --save_stats --plot_spectra
python3 evaluate_es.py --agent_name=$1 --env_name=$ENV13 --save_plots --use_best_elite --save_trajectory --save_stats --plot_spectra