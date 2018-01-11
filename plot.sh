DATE=$(date '+%Y%m%d_%H:%M')
grep "^Score" $(ls -t logs* | head -1) | sed 's/Score: //; s/Epoch:.*//' > scores.txt
gnuplot -e 'set term png; set output "wykres_'$DATE'.png"; f(x) = a*x + b; fit f(x) "scores.txt" via a, b; plot f(x), "scores.txt"'

