%% 训练好的SVM最终的决策函数
function plabel = DecisionFunction(x,model) 
gamma = model.Parameters(4); 
RBF = @(u,v)( exp(-gamma.*sum( (u-v).^2) ) ); 
len = length(model.sv_coef); 
y = 0; 
for i = 1:len 
    u = model.SVs(i,:); 
    y = y + model.sv_coef(i)*RBF(u,x); 
end
b = -model.rho; 
y = y + b; 
if y >= 0 
    plabel = 1; 
else
    plabel = -1;
end