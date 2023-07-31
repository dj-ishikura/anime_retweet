#!/bin/bash

# Get the filename from the arguments
filename="$1"

awk -F, '{
  cmd="echo $3 | jq -r .created_at"
  cmd | getline d
  close(cmd)
  print d","$0
}' "$filename" | sort -t, -k1,1 -r | awk -F, '!a[$2]++ {print $2","$3","$4}'
