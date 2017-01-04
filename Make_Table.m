columnName =   {'Code','Dev. Time', 'Comp. Time', 'Speed-up', 'Price'};
data =  {
    'Matlab 1 core', '2 hours',sprintf('%1.1f',time(1)),sprintf('%1.1f',1.0),'~300$';
    'Matlab 20 cores', '2 hours',sprintf('%1.1f',time(2)), sprintf('%1.1f',time(1)/time(2)), '~10 000$';
    'C 1-core', '5 hours',sprintf('%1.1f',time(3)), sprintf('%1.1f',time(1)/time(3)), '~500$';    
    'C 20-codes', '5.5 hours',sprintf('%1.1f',time(4)), sprintf('%1.1f',time(1)/time(4)), '~10 000$'; 
    'Matlab GPU DOUBLE', '2 hours',sprintf('%1.1f',time(6)), sprintf('%1.1f',time(1)/time(5)), '~2000$';
    'Matlab GPU SINGLE', '2 hours',sprintf('%1.1f',time(5)), sprintf('%1.1f',time(1)/time(6)), '~2000$';
    'CUDA C DOUBLE', '4 days',sprintf('%1.1f',time(8)), sprintf('%1.1f',time(1)/time(8)), '~2000$';
    'CUDA C SINGLE', '4 days',sprintf('%1.1f',time(7)), sprintf('%1.1f',time(1)/time(7)), '~2000$';    
};

dataSize = size(data);
maxLen = zeros(1,dataSize(2));
for i=1:dataSize(2)
      for j=1:dataSize(1)
          len = length(data{j,i});
          if(len > maxLen(1,i))
              maxLen(1,i) = len;
          end
      end
end
maxLen=max(maxLen,11);
cellMaxLen = num2cell(maxLen*15);

fh = figure;
t = uitable('Units','normalized','Position', [0 0 1 1], 'Data', data, 'ColumnName', columnName,'RowName',[]);
figPos = get(fh,'Position');
t.FontSize=14;
set(t, 'ColumnWidth', cellMaxLen);
tableExtent = get(t,'Extent');
set(fh,'Position',[figPos(1:2), 1.05*figPos(3:4).*tableExtent(3:4)]);
