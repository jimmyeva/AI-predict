#!/bin/sh

pythonPath=/home/bi/algorithm/salesForecasting/runPython/
logPath=/home/bi/algorithm/salesForecasting/runLog/
runJarName="cj_prediction.py"

date=`date +%Y-%m-%d `
currPath=$logPath$date
logFile=$date".log"


currFile=$currPath"/"$logFile

if [ ! -x "$currPath" ]; then
    mkdir $currPath
fi

if [ ! -f "$currFile" ]; then
   touch $currFile
fi


startTime=$1

echo "------------------------------开始--`date +%Y-%m-%d,%H-%M-%S`----------------------------"  >>$currFile
python $pythonPath$runJarName $startTime  >>$currFile


runResult=`tail -1 $currFile`

echo "------------------------------结束--`date +%Y-%m-%d,%H-%M-%S`----------------------------"  >>$currFile

echo "runResult[$runResult]"
strindex=`expr index $runResult ':'`
resultStatus=${runResult:${strindex}:1}


if [ $resultStatus -eq 1 ]; then
  exit 1
else
  exit 0
fi

#echo $status
