#!/bin/bash

for i in {2089357..2089370}; do
  qdel -Wforce $i
done
