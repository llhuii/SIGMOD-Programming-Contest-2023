input=${1:-dummy-output.bin}
gt=${2:-${input/m./m-gt.}}
{
echo ${title:-new} ==========
date
time ./knng "$input"  && evaluating -truth "$gt" -eval output.bin
date
} 2>&1| tee -a "logs/$(basename $input)-$debug.txt"
