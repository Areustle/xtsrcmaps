local dap = require('dap')
-- local mason_registry = require("mason-registry")

dap.adapters.codelldb = {
    type = 'server',
    port = '${port}',
    executable = {
        command = vim.fn.join({ vim.fn.stdpath "data", "mason/bin/codelldb" }, "/"),
        -- command = mason_registry.get_package("codelldb"),
        args = { "--port", "${port}" },
    },
    name = 'codelldb',
}

dap.configurations.cpp = {
    {
        name = 'gtsrcmaps_debug',
        type = 'codelldb',
        request = 'launch',
        program = function()
            return vim.fn.input('Path to executable: ', vim.fn.getcwd() .. '/', 'file')
        end,
        -- program = function()
        --     return vim.fn.getcwd() .. '/Debug/Likelihood/gtsrcmaps'
        --     -- return vim.fn.join({vim.fn.getcwd(), 'Debug', 'Likelihood', 'gtsrcmaps'}, '/')
        -- end,
        cwd = '${workspaceFolder}',
        stopOnEntry = false,
    },
    -- {
    --     name = 'gtsrcmaps_rwdi',
    --     type = 'codelldb',
    --     request = 'launch',
    --     program = function()
    --         return vim.fn.getcwd() .. '/RelWithDebInfo/Likelihood/gtsrcmaps'
    --     end,
    --     -- program = 'RelWithDebInfo/Likelihood/gtsrcmaps',
    --     cwd = '${workspaceFolder}',
    --     stopOnEntry = false,
    --     args = {},
    -- },
}

dap.configurations.c = dap.configurations.cpp
