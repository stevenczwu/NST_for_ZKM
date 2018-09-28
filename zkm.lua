require 'torch'
require 'nn'
require 'image'
require 'camera'

require 'qt'
require 'qttorch'
require 'qtwidget'

require 'fast_neural_style.ShaveImage'
require 'fast_neural_style.TotalVariation'
require 'fast_neural_style.InstanceNormalization'
local utils = require 'fast_neural_style.utils'
local preprocess = require 'fast_neural_style.preprocess'

local cmd = torch.CmdLine()

-- Model options
cmd:option('-models', 'trained_models/phoenix.t7,trained_models/sketch.t7,trained_models/chemie2.t7')
cmd:option('-height', 896) 
cmd:option('-width', 1600)
cmd:option('-style_images', 'style_images/phoenix.jpg,style_images/sketch.jpg,style_images/chemie.jpg')
cmd:option('-show_style_images', 1)

-- GPU options
cmd:option('-gpu', 0)
cmd:option('-backend', 'cuda')
cmd:option('-use_cudnn', 1)

-- Webcam options
cmd:option('-webcam_idx', 0)
cmd:option('-webcam_fps', 60)
cmd:option('-portrait_mode', 1)
cmd:option('-mirror_mode', 1)


local function shuffle(array)
    -- do fisher-yates shuffle 
    local output = { }
    math.randomseed(os.time())
    local random = math.random

    for index = 1, #array do
        local offset = index - 1
        local value = array[index]
        local randomIndex = offset*random()
        local flooredIndex = randomIndex - randomIndex%1
        if flooredIndex == offset then
            output[#output + 1] = value
        else
            output[#output + 1] = output[flooredIndex + 1]
            output[flooredIndex + 1] = value
        end
    end

    return output
end


local function main()
    local opt = cmd:parse(arg)
    local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
    local models = {}
    local style_path = {}
    local styleimages = {}
    local preprocess_method = nil

    -- Load style images
    for _, style_image_path in ipairs(opt.style_images:split(',')) do
        table.insert(style_path, style_image_path)
    end
    for _, path in ipairs(style_path) do
        local img = image.load(path, 3)
        table.insert(styleimages, img)
    end

    -- construct a white image
    local white = qt.QImage.fromTensor(torch.ones(3, 1, 1))

    -- Load models
    for _, checkpoint_path in ipairs(opt.models:split(',')) do
        print('loading model from ', checkpoint_path)
        local checkpoint = torch.load(checkpoint_path)
        local model = checkpoint.model
    
        model:evaluate()
        model:type(dtype)
        if use_cudnn then
            cudnn.convert(model, cudnn)
        end
        table.insert(models, model)
        local this_preprocess_method = checkpoint.opt.preprocessing or 'vgg'
        if not preprocess_method then
            preprocess_method = this_preprocess_method
            print('Preprocess method: ' .. preprocess_method)
        else
            if this_preprocess_method ~= preprocess_method then
                error('All models must use the same preprocessing')
            end
        end
    end

    local preprocess = preprocess[preprocess_method]

    -- Initialize camera
    local camera_opt = {
        idx = opt.webcam_idx,
        fps = opt.webcam_fps,
        height = opt.height,
        width = opt.width,
        }
    local cam = image.Camera(camera_opt)

    -- Initialize random orders 
    local order_last = {}
    local order_new = {}
    for i = 1, #(style_path) do
        table.insert(order_last,i)
    end
    order_new = shuffle(order_last)
    if #(order_last) > 1 then
        while order_new[1] == order_last[#(order_last)] do
            order_new = shuffle(order_last)
        end
    end
    order_last = order_new

    local win = nil
    local idx = 1
    local time = 8
    local init_time = os.time() 
    local use_model_idx = order_new[idx]
    local model = models[use_model_idx]
    local style = styleimages[use_model_idx]
    print('Total number of models: ' .. #order_last)
    print('===== Start Style Transfer =====')
    print('Using model: No.' .. use_model_idx)

  -- ==========================================
  --                main loop
  -- ========================================== 
    while true do

        -- Change model after time seconds ********************
        local diff_time = os.difftime(os.time(),init_time)
        if diff_time > time then
            idx = idx + 1
            init_time = os.time()
            if idx > #(style_path) then
                -- Create a new random order for styles
                order_new = shuffle(order_last)
                if #(order_last) > 1 then
                    while order_new[1] == order_last[#(order_last)] do
                        order_new = shuffle(order_last)
                    end
		end
                order_last = order_new
                print('======= A new round! =======')
                idx = 1   
            end
            model:clearState()
            use_model_idx = order_new[idx]
            print('Using model: No.' .. use_model_idx)
            model = models[use_model_idx]
            style = styleimages[use_model_idx]
        end

        -- Grab a frame from the webcam ***********************
        local img = cam:forward()
        -- rotate image to portrait mode 
        if opt.portrait_mode then 
            local img_rotated = img:transpose(2, 3)
            img = img_rotated:contiguous()
            img = image.vflip(img)
        end
        -- flip image to mirror mode
        if opt.mirror_mode then
            img = image.hflip(img)
        end

        -- Preprocess the frame
        local H, W = img:size(2), img:size(3)
        img = img:view(1, 3, H, W)
        local img_pre = preprocess.preprocess(img):type(dtype)

        -- Run the models *************************************
        local img_out_pre = model:forward(img_pre)

        -- Deprocess the frame
        local img_out = preprocess.deprocess(img_out_pre)[1]:float()
        local img_disp = img_out
	
        -- Display result image *******************************
        if not win then
            -- On the first call, construct a window
            win = qtwidget.newwindow(W,H)
            win.widget:window():showFullScreen()
        else
            local size = win.widget:window().size:totable()
            local WxH = size.width .. 'x' .. size.height
            local img_disp_scale = image.scale(img_disp, WxH)
            image.display{image=img_disp_scale, win=win, min=0, max=1, gui=false, zoom=0, nrow=1}

            -- Show style images or not
            if opt.show_style_images then
	        local stylew = size.width / 4
                local styleh = size.height / 10
	        local qt_img_style = qt.QImage.fromTensor(style)
                win.port:image(size.width-stylew-25, size.height-styleh-25, stylew+10, styleh+10, white)
                win.port:image(size.width-stylew-20, size.height-styleh-20, stylew, styleh, qt_img_style)
            end
        end
    end
end


main()

