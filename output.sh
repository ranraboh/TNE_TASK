#!/bin/bash
output_log=./out/tne.out
errors_log=./out/tne.err

include_errors="no"
rm_logs="no"

numargs=$#
if [[ $numargs > 0 ]]; then
  if [[ $1 == "--include-errors" ]]; then
    include_errors="yes"
  elif [[ $1 == "--rm-logs" ]]; then
    rm_logs="yes"
  fi
fi

if [[ $rm_logs == "yes" ]]; then
  rm -rf ./out/*
  exit 0
fi

echo "Checking output"
if [[ ! -f $output_log ]]; then
  echo "Still running.."
  echo "Output log:"
  cat ./tne_ran.out
  echo "Queue:"
  squeue -u "ranraboh"
  exit 0
fi
cat $output_log
if [[ $include_errors == "yes" ]]; then
  echo "Errors log:"
  cat $errors_log
fi