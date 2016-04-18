require 'nn'

local function orthogonal_init(shape)
    -- add checking for <2 dimensionality as they do in lasagne
    local m = 1
    for i=2,shape:size() do
        m=m*shape[i]
    end
    f_shape = torch.LongStorage({shape[1], m})
    a = torch.randn(f_shape)
    u, s, v = torch.svd(a)
    -- check this later, in numpy they use 'full_matrices' arg
    -- in torch svd returns square u,v matrices
    if f_shape[1] > f_shape[2] then
        q = u[{{},{1, f_shape[2]}}]
    else
        q = v[{{1, f_shape[1]},{}}]
    end
    q = q:reshape(shape)
    return q
end

local function lsuv_init(model, batch, tol_var, t_max)
    local tol_var = tol_var or 0.1
    local t_max   = t_max or 10

  	for i=1,#model.modules do
		    local m = model.modules[i]
        if m.weight == nil then
          break
        end
        local w = orthogonal_init(m.weight:size())
        m.weight = w
        local out = m:updateOutput(batch)
        local var = torch.var(out)
        local t_i = 1
        while(torch.abs(var - 1.0) >= tol_var and t_i < t_max) do
            w = w / var
            m.weight = w
            var = torch.var(m:updateOutput(batch))
            t_i = t_i + 1
        end
    end
    return model
end

return lsuv_init
