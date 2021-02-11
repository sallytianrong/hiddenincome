function [c,d,Q] = hold_u_v_constant(u,v,y,a1,a2,price,startval)
% This function inputs both agents' utilities and solves allocation

    % Turn off fmincon display
    options = optimoptions('fmincon','Display','off');
    % d: consumption of agent 2
    d_zero = @(d) abs((1-a1).*log(y - exp(u./a1 - (1-a1).*v./(a1.*(1-a2)) + (1-a1).*a2.*log(d)./(a1.*(1-a2))) - d)...
    + (1-a1).*a2.*log(d)./(1-a2) - (1-a1).*v./(1-a2) - (1-a1).*log(price));
    d = fmincon(d_zero, startval,[],[],[],[],0,y,[],options);
    % c: consumption of agent 1, Q: public good consumption
    c = max(exp(u./a1 - (1-a1).*v./(a1.*(1-a2)) + (1-a1).*a2.*log(d)./(a1.*(1-a2))),0.01);
    Q = (y - c - d)./price;
    % corner solution
    if Q < 0
        Q = 0.01;
        denom = c+d;
        c = c.*(y-price.*Q)./denom;
        d = c.*(y-price.*Q)./denom;
    end