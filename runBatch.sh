wait_exec ()
{
    while pidof knng; do
        sleep 7;
    done;
    make && ( bash /root/cpp/evaluate.sh $1 )
}

do_run() {
  local retcode=0
  echo ====testing batch $ITER $L $R
  date
  wait_exec ~/contest-release-10m.bin  2>&1| awk -v L=$L -v R=$R -v ITER=$ITER '1;
  /^Build time:/{a+=$3>1650;total=$3}
  END{
  print "total: L="L, "R="R, "iter="ITER, total;
  exit(a)
  }' || $retcode=1
  echo 3 > /proc/sys/vm/drop_caches
  sleep 60
  date
  return $retcode
}

export NCONTROLS=0 ITER L R
# last L:185, R:540  iter=7
{
echo ====

for ITER in ; do
for L in {180..199..5}; do
  for R in {500..665..40}; do
  do_run || break
  done
done
done

for ITER in 7 8; do
for L in {200..231..5}; do
  for R in {350..459..40}; do
  do_run || break
  done
done
done

} | tee -a batch.log
