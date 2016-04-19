require 'nn'

local function lsuv_init(model, get_batch, tol_var, t_max)
   local tol_var = tol_var or 0.1
   local t_max   = t_max or 10

   for i=1,#model.modules do
      local m = model.modules[i]
      if m.weight ~= nil then
         local t_i = 1
         while true do
            local input = get_batch()
            model:forward(input)
            local out = m.output
            local var = torch.var(out)
            if torch.abs(var - 1.0) < tol_var or t_i > t_max then
               break
            end
            m.weight:div(math.sqrt(var))
         end
      end
   end
   return model
end

return lsuv_init
