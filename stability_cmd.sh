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

python3 agent_stability.py --env_name=$ENV1 --agent_name=$1 --use_best_mu
python3 agent_stability.py --env_name=$ENV1 --agent_name=$1 --use_mu
python3 agent_stability.py --env_name=$ENV1 --agent_name=$1 --use_best_elite
python3 agent_stability.py --env_name=$ENV1 --agent_name=$1

python3 agent_stability.py --env_name=$ENV2 --agent_name=$1 --use_best_mu
python3 agent_stability.py --env_name=$ENV2 --agent_name=$1 --use_mu
python3 agent_stability.py --env_name=$ENV2 --agent_name=$1 --use_best_elite
python3 agent_stability.py --env_name=$ENV2 --agent_name=$1

python3 agent_stability.py --env_name=$ENV3 --agent_name=$1 --use_best_mu
python3 agent_stability.py --env_name=$ENV3 --agent_name=$1 --use_mu
python3 agent_stability.py --env_name=$ENV3 --agent_name=$1 --use_best_elite
python3 agent_stability.py --env_name=$ENV3 --agent_name=$1

python3 agent_stability.py --env_name=$ENV4 --agent_name=$1 --use_best_mu
python3 agent_stability.py --env_name=$ENV4 --agent_name=$1 --use_mu
python3 agent_stability.py --env_name=$ENV4 --agent_name=$1 --use_best_elite
python3 agent_stability.py --env_name=$ENV4 --agent_name=$1

python3 agent_stability.py --env_name=$ENV5 --agent_name=$1 --use_best_mu
python3 agent_stability.py --env_name=$ENV5 --agent_name=$1 --use_mu
python3 agent_stability.py --env_name=$ENV5 --agent_name=$1 --use_best_elite
python3 agent_stability.py --env_name=$ENV5 --agent_name=$1

python3 agent_stability.py --env_name=$ENV6 --agent_name=$1 --use_best_mu
python3 agent_stability.py --env_name=$ENV6 --agent_name=$1 --use_mu
python3 agent_stability.py --env_name=$ENV6 --agent_name=$1 --use_best_elite
python3 agent_stability.py --env_name=$ENV6 --agent_name=$1

python3 agent_stability.py --env_name=$ENV7 --agent_name=$1 --use_best_mu
python3 agent_stability.py --env_name=$ENV7 --agent_name=$1 --use_mu
python3 agent_stability.py --env_name=$ENV7 --agent_name=$1 --use_best_elite
python3 agent_stability.py --env_name=$ENV7 --agent_name=$1

python3 agent_stability.py --env_name=$ENV8 --agent_name=$1 --use_best_mu
python3 agent_stability.py --env_name=$ENV8 --agent_name=$1 --use_mu
python3 agent_stability.py --env_name=$ENV8 --agent_name=$1 --use_best_elite
python3 agent_stability.py --env_name=$ENV8 --agent_name=$1

python3 agent_stability.py --env_name=$ENV9 --agent_name=$1 --use_best_mu
python3 agent_stability.py --env_name=$ENV9 --agent_name=$1 --use_mu
python3 agent_stability.py --env_name=$ENV9 --agent_name=$1 --use_best_elite
python3 agent_stability.py --env_name=$ENV9 --agent_name=$1

python3 agent_stability.py --env_name=$ENV10 --agent_name=$1 --use_best_mu
python3 agent_stability.py --env_name=$ENV10 --agent_name=$1 --use_mu
python3 agent_stability.py --env_name=$ENV10 --agent_name=$1 --use_best_elite
python3 agent_stability.py --env_name=$ENV10 --agent_name=$1

python3 agent_stability.py --env_name=$ENV11 --agent_name=$1 --use_best_mu
python3 agent_stability.py --env_name=$ENV11 --agent_name=$1 --use_mu
python3 agent_stability.py --env_name=$ENV11 --agent_name=$1 --use_best_elite
python3 agent_stability.py --env_name=$ENV11 --agent_name=$1

python3 agent_stability.py --env_name=$ENV12 --agent_name=$1 --use_best_mu
python3 agent_stability.py --env_name=$ENV12 --agent_name=$1 --use_mu
python3 agent_stability.py --env_name=$ENV12 --agent_name=$1 --use_best_elite
python3 agent_stability.py --env_name=$ENV12 --agent_name=$1

python3 agent_stability.py --env_name=$ENV13 --agent_name=$1 --use_best_mu
python3 agent_stability.py --env_name=$ENV13 --agent_name=$1 --use_mu
python3 agent_stability.py --env_name=$ENV13 --agent_name=$1 --use_best_elite
python3 agent_stability.py --env_name=$ENV13 --agent_name=$1