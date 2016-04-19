require 'nn'

local function lsuv_init(model, batch, tol_var, t_max)
    local tol_var = tol_var or 0.1
    local t_max   = t_max or 10

  	for i=1,#model.modules do
		    local m = model.modules[i]
        if m.weight == nil then
          m:updateOutput(model.modules[i-1].output)
        else
          local input = (i > 1 and batch) or model.modules[i-1].output
          local out = m:updateOutput(input)
          local var = torch.var(out)
          local t_i = 1
          while(torch.abs(var - 1.0) >= tol_var and t_i < t_max) do
            m.weight:cdiv(var)
            var = torch.var(m:updateOutput(input))
            t_i = t_i + 1
          end
        end
    end
    return model
end

return lsuv_init
