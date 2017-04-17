#!/usr/bin/env bash
arctype=$1
matchcode=$2
getcode=$3
time zcat /om/user/futrell/syntngrams/$arctype.*-of-99.gz  | python3 syntngrams_depmi.py $matchcode $getcode | sort | sh uniqsum.sh > /om/user/futrell/arcs_$matchcode-$getcode
