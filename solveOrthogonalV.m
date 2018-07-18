function [ V ] = solveOrthogonalV( B, X, inv_Xcov )

n = size(X, 1);
L = size(B, 2);
d = size(X, 2);

V = zeros(d, L);
XtB = X'*B;                                             % d*n*L
inv_XcovV = zeros(d, L);

A = zeros(L, L);
b = zeros(L, 1);

for k=1:L  
    for i=1:k-1
        A(k - 1, i) = V(:, k-1)'*inv_XcovV(:, i);       % d
        A(i, k - 1) = A(k - 1, i);
        b(i) = XtB(:, k)'*inv_XcovV(:, i);              % d
    end
    if (k > 1)
        phi = A(1:k - 1, 1:k - 1)\b(1:k - 1);
    else
        phi = [];
    end
    tmp  = 0;
    for i=1:length(phi)
        tmp = tmp + phi(i)*V(:, i);
    end
    V(:, k) = inv_Xcov*(XtB(:, k) - tmp);               % d*d
    inv_XcovV(:, k) = (inv_Xcov*V(:, k));               % d*d
end
end

