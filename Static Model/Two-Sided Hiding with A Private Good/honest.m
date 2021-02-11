function y = honest(x,alpha,delta1,delta2,inc,rho,prob)

% consumption of agent 1
c1 = zeros(size(prob));
c1(:,1) = [x;x+1-delta2;x+2-2*delta2];
c1 =[c1(:,1) c1(:,1)+delta1.*rho c1(:,1)+2*delta1.*rho];
d1 = prob.*sqrt(c1);
% consumption of agent 2
c2 = inc - c1;
d2 = prob.*sqrt(c2);
% utility
y = - sum(sum(alpha.*d1 + (1-alpha).*d2),2);

end



