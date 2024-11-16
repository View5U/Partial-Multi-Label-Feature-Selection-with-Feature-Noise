function B = Constraint01(A)
B = zeros(size(A));
for i = 1:size(A, 1)
    for j = 1:size(A, 2)
        if A(i, j) > 1
            B(i, j) = 1;
        elseif A(i, j) < 0
            B(i, j) = 0;
        else
            B(i, j) = A(i, j);
        end
    end
end
end
