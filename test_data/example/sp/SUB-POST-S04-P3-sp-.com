%nprocshared=4
%mem=300MW
#p def2tzvpp um062x pop=(full,nboread) stable=opt

SUB-POST-S04-P3-sp-

-1 1
 C                  3.26585400   -0.90324400   -0.00526300
 C                  2.16493200    0.12876600    0.02344500
 H                  4.23300600   -0.40209500    0.02642400
 H                  3.16789200   -1.57911600    0.84985100
 H                  3.18862400   -1.51023700   -0.91244300
 C                 -0.19836700    0.43264800   -0.00060900
 H                 -0.17013200    1.01913000    0.92256000
 H                 -0.12618100    1.12752600   -0.84356300
 C                 -1.45241000   -0.42806300   -0.08592300
 C                 -2.70481200    0.38807400   -0.13297200
 H                 -1.48006100   -1.11896200    0.76865900
 C                 -4.03678200   -0.22422800    0.14630400
 H                 -2.66790700    1.36787100   -0.60576800
 H                 -4.36367200   -0.89345600   -0.66983400
 H                 -4.01569900   -0.84011500    1.05571600
 O                  2.32834900    1.33066100    0.08504300
 O                  0.94542200   -0.45536000   -0.02839700
 H                 -4.81601300    0.53477900    0.26706100
 H                 -1.37050900   -1.07144900   -0.98172700

$nbo bndidx $end