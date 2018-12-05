function R = standardDeviation(x)
%STANDARDDEVIATION Computes standard deviation

R = sqrt(sum((x - mean(x)) .** 2)/(length(x)-1));

end