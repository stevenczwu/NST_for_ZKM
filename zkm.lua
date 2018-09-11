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
cmd:option('-models', 'trained_models/phoenix.t7')
cmd:option('-height', 1000) 
cmd:option('-width', 1600)
cmd:option('-style_images', 'style_images/phoenix.jpg')
cmd:option('-show_style_images', 1)

-- GPU options
cmd:option('-gpu', -1)
cmd:option('-backend', 'cuda')
cmd:option('-use_cudnn', 1)

-- Webcam options
cmd:option('-webcam_idx', 0)
cmd:option('-webcam_fps', 60)
cmd:option('-portrait_mode', 1)
cmd:option('-mirror_mode', 0)

local function main()
  local opt = cmd:parse(arg)

  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
  local models = {}
  local styleimages = {}
  local style_path = {}
  local preprocess_method = nil
  local i = 1

  -- Load style images
  for _, style_image_path in ipairs(opt.style_images:split(',')) do
    table.insert(style_path, style_image_path)
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
    local img = image.load(style_path[i], 3)
    i = i + 1
    table.insert(styleimages, img)
    local this_preprocess_method = checkpoint.opt.preprocessing or 'vgg'
    if not preprocess_method then
      print('got here')
      preprocess_method = this_preprocess_method
      print(preprocess_method)
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

  local win = nil
  local idx = 1
  local time = 8
  local init = os.time() 
  local model = models[idx]
  local style = styleimages[idx]

  -- ==========================================
  --                main loop
  -- ========================================== 
  while true do
    -- Change model after time seconds ********************
    local diff=os.difftime(os.time(),init)
    if diff > time then
      idx = idx + 1
      init = os.time()
      if idx > table.getn(models) then
         idx = 1   
      end
      model:clearState()
      model = models[idx]
      style = styleimages[idx]
    end

    -- Grab a frame from the webcam ***********************
    local img = cam:forward()
    -- rotate image to portrait mode 
    if opt.portrait_mode == 1 then 
        local img_rotated = img:transpose(2, 3)
        img = img_rotated:contiguous()
        img = image.vflip(img)
    end
    -- flip image to mirror mode
    if opt.mirror_mode == 1 then
        img = image.hflip(img)
    end
    -- Preprocess the frame
    local H, W = img:size(2), img:size(3)
    img = img:view(1, 3, H, W)
    local img_pre = preprocess.preprocess(img):type(dtype)

    -- Run the models *************************************
    local imgs_out = {}   
    local img_out_pre = model:forward(img_pre)
    -- Deprocess the frame
    local img_out = preprocess.deprocess(img_out_pre)[1]:float()
    table.insert(imgs_out, img_out)
    local img_disp = image.toDisplayTensor{
      input = imgs_out,
      min = 0,
      max = 1,
      nrow = math.floor(math.sqrt(#imgs_out)),
    }
	
    -- Display result image *******************************
    if not win then
      -- On the first call use image.display to construct a window
      win = image.display(img_disp)
      win.window:showFullScreen()
    else
      -- Reuse the same window
      win.image = img_out
      local size = win.window.size:totable()
      local qt_img = qt.QImage.fromTensor(img_disp)
      local qt_img_style = qt.QImage.fromTensor(style)
      win.painter:image(0, 0, size.width, size.height, qt_img)

      -- Show style images or not
      if opt.show_style_images == 1 then
        local stylew = size.width / 4
        local styleh = size.height / 10
        win.painter:image(size.width - stylew - 25  , size.height - styleh - 25, stylew+10, styleh+10, white)
        win.painter:image(size.width - stylew - 20  , size.height - styleh - 20, stylew, styleh, qt_img_style)
      end
    end
  end
end


main()

