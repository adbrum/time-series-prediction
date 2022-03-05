#!/bin/bash
#sync
#echo "echo 3 > /proc/sys/vm/drop_caches"
sudo sync && sudo sysctl vm.drop_caches=3
