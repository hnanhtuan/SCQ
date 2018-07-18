function [ V ] = solveOrthonormalV( B, X, U, S, init_V )

n = size(X, 1);
L = size(B, 2);
d = size(X, 2);

V = zeros(d, L);
XtB = X'*B;                                                 % d*n*L

for k=1:L
    tmp = 0;
    Ac = cell(L, L);
    bc = cell(L, 1);
    pre_err = [];
    counter = 0;
    while (true)
    % for iter=1:200
        lambda = solveLambda(-S(1), 1000000, (U'*(XtB(:, k) - tmp)).^2, S);
        if (lambda == 0)
            V = init_V;
            return;
        end
        
        inv_Xcov = U*diag(1./(S + lambda))*U';              % d^3
        for c=1:k-1
            inv_XcovV = inv_Xcov*V(:, c);                   % d*d
            for r=1:k-1
                Ac{r, c} = V(:, r)'*inv_XcovV;              % d
%                 Ac{c, r} = Ac{r, c};
            end
            bc{c} = XtB(:, k)'*inv_XcovV;                   % d
        end
        tmp = 0;
        if (k > 1)
            A = cell2mat(Ac(1:k - 1, 1:k - 1));
            b = cell2mat(bc(1:k - 1));
            phi = A(1:k - 1, 1:k - 1)\b(1:k - 1);           % L^3
            tmp = sum(repmat(phi', size(V, 1), 1).*V(:, 1:length(phi)), 2);
        end
        
        V(:, k) = inv_Xcov*(XtB(:, k) - tmp);               % d*d
        
        err = abs(V(:, k)'*V(:, k) - 1);
        if (err < 1e-3)
            break;
        end
        if ((counter > 2000) && (err < 1e-2))
            break;
        end
        
        if (abs(pre_err - err) < 1e-8)
            V = init_V;
            return;
        end
        pre_err = err;
        counter = counter + 1;
    end
end
end

function [lambda] = solveLambda(low, high, VtXtBk2, S)
pre_gap = -1;
while (low < high)
% for iter=1:1000    
    mid = low + ( high - low )/2;
    val = sum(VtXtBk2./((S + mid).^2));
    gap = high - low;
    if (abs(val - 1) < 1e-4) 
        break; 
    end;
    if  (pre_gap == gap)
        lambda = 0;
        return;
    end;
    if (val > 1)
        low = mid;
    else
        high = mid;
    end
    pre_gap = gap;
end
lambda = mid;
end
