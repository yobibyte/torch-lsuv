-- LSUV initialisation from this paper http://arxiv.org/abs/1511.06422

require 'nn'

local function lsuv_init(model, get_batch, tol_var, t_max)
   local tol_var = tol_var or 0.1
   local t_max   = t_max or 10

   for _,m in ipairs(model:listModules()) do
      if m.weight ~= nil then
         local t_i = 1
         while true do
            local input = get_batch()
            model:forward(input)
            local var = torch.var(m.output)
            if torch.abs(var - 1.0) < tol_var or t_i > t_max then
               break
            end
            m.weight:div(math.sqrt(var))
            t_i = t_i + 1
         end
      end
   end
   return model
end

return lsuv_init
