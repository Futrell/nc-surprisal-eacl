#!/usr/bin/env bash
awk -F"\t" '{if (v != $1) { print v"\t"c; c=$2; v=$1 } else c+=$2}' | sed '1d'
