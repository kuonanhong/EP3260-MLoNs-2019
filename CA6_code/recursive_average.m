function [out_avg, avg_index] = recursive_average(out_avg, input_instant, avg_index)

avg_index = avg_index + 1;
switch 1
    case any(avg_index == 1)
        out_avg = input_instant;
    otherwise
        try
            out_avg       = out_avg +  bsxfun(@times, (1./avg_index), (input_instant - out_avg));            
        catch
           out_avg       = out_avg +  bsxfun(@times, (1./avg_index'), (input_instant - out_avg));
        end
end