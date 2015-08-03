load conv.dat;

plot(conv(:,1), conv(:,2), '*-')
xlabel('K')
ylabel('L2 error')
title('L2error vs K')


