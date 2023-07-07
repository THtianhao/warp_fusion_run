
def do_run():
    seed = args.seed
    print(range(args.start_frame, args.max_frames))
    if args.animation_mode != "None":
        batchBar = tqdm(total=args.max_frames, desc="Frames")

    # if (args.animation_mode == 'Video Input') and (args.midas_weight > 0.0):
    # midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = init_midas_depth_model(args.midas_depth_model)
    for frame_num in range(args.start_frame, args.max_frames):
        if stop_on_next_loop:
            break

        # display.clear_output(wait=True)

        # Print Frame progress if animation mode is on
        if args.animation_mode != "None":
            display.display(batchBar.container)
            batchBar.n = frame_num
            batchBar.update(1)
            batchBar.refresh()
            # display.display(batchBar.container)

        # Inits if not video frames
        if args.animation_mode != "Video Input Legacy":
            if args.init_image == '':
                init_image = None
            else:
                init_image = args.init_image
            init_scale = get_scheduled_arg(frame_num, init_scale_schedule)
            # init_scale = args.init_scale
            steps = int(get_scheduled_arg(frame_num, steps_schedule))
            style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
            skip_steps = int(steps - steps * style_strength)
            # skip_steps = args.skip_steps

        if args.animation_mode == 'Video Input':
            if frame_num == args.start_frame:
                steps = int(get_scheduled_arg(frame_num, steps_schedule))
                style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
                skip_steps = int(steps - steps * style_strength)
                # skip_steps = args.skip_steps

                # init_scale = args.init_scale
                init_scale = get_scheduled_arg(frame_num, init_scale_schedule)
                # init_latent_scale = args.init_latent_scale
                init_latent_scale = get_scheduled_arg(frame_num, latent_scale_schedule)
                init_image = f'{videoFramesFolder}/{frame_num + 1:06}.jpg'
                if use_background_mask:
                    init_image_pil = Image.open(init_image)
                    init_image_pil = apply_mask(init_image_pil, frame_num, background, background_source, invert_mask)
                    init_image_pil.save(f'init_alpha_{frame_num}.png')
                    init_image = f'init_alpha_{frame_num}.png'
                if (args.init_image != '') and args.init_image is not None:
                    init_image = args.init_image
                    if use_background_mask:
                        init_image_pil = Image.open(init_image)
                        init_image_pil = apply_mask(init_image_pil, frame_num, background, background_source, invert_mask)
                        init_image_pil.save(f'init_alpha_{frame_num}.png')
                        init_image = f'init_alpha_{frame_num}.png'
                if VERBOSE: print('init image', args.init_image)
            if frame_num > 0 and frame_num != frame_range[0]:
                # print(frame_num)

                first_frame_source = batchFolder + f"/{batch_name}({batchNum})_{args.start_frame:06}.png"
                if os.path.exists(first_frame_source):
                    first_frame = Image.open(first_frame_source)
                else:
                    first_frame_source = batchFolder + f"/{batch_name}({batchNum})_{args.start_frame - 1:06}.png"
                    first_frame = Image.open(first_frame_source)

                # print(frame_num)

                # first_frame = Image.open(batchFolder+f"/{batch_name}({batchNum})_{args.start_frame:06}.png")
                # first_frame_source = batchFolder+f"/{batch_name}({batchNum})_{args.start_frame:06}.png"
                if not fixed_seed:
                    seed += 1
                if resume_run and frame_num == start_frame:
                    print('if resume_run and frame_num == start_frame')
                    img_filepath = batchFolder + f"/{batch_name}({batchNum})_{start_frame - 1:06}.png"
                    if turbo_mode and frame_num > turbo_preroll:
                        shutil.copyfile(img_filepath, 'oldFrameScaled.png')
                    else:
                        shutil.copyfile(img_filepath, 'prevFrame.png')
                else:
                    # img_filepath = '/content/prevFrame.png' if is_colab else 'prevFrame.png'
                    img_filepath = 'prevFrame.png'

                next_step_pil = do_3d_step(img_filepath, frame_num, forward_clip=forward_weights_clip)
                if warp_mode == 'use_image':
                    next_step_pil.save('prevFrameScaled.png')
                else:
                    # init_image = 'prevFrameScaled_lat.pt'
                    # next_step_pil.save('prevFrameScaled.png')
                    torch.save(next_step_pil, 'prevFrameScaled_lat.pt')

                steps = int(get_scheduled_arg(frame_num, steps_schedule))
                style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
                skip_steps = int(steps - steps * style_strength)
                # skip_steps = args.calc_frames_skip_steps

                ### Turbo mode - skip some diffusions, use 3d morph for clarity and to save time
                if turbo_mode:
                    if frame_num == turbo_preroll:  # start tracking oldframe
                        if warp_mode == 'use_image':
                            next_step_pil.save('oldFrameScaled.png')  # stash for later blending
                        if warp_mode == 'use_latent':
                            # lat_from_img = get_lat/_from_pil(next_step_pil)
                            torch.save(next_step_pil, 'oldFrameScaled_lat.pt')
                    elif frame_num > turbo_preroll:
                        # set up 2 warped image sequences, old & new, to blend toward new diff image
                        if warp_mode == 'use_image':
                            old_frame = do_3d_step('oldFrameScaled.png', frame_num, forward_clip=forward_weights_clip_turbo_step)
                            old_frame.save('oldFrameScaled.png')
                        if warp_mode == 'use_latent':
                            old_frame = do_3d_step('oldFrameScaled.png', frame_num, forward_clip=forward_weights_clip_turbo_step)

                            # lat_from_img = get_lat_from_pil(old_frame)
                            torch.save(old_frame, 'oldFrameScaled_lat.pt')
                        if frame_num % int(turbo_steps) != 0:
                            print('turbo skip this frame: skipping clip diffusion steps')
                            filename = f'{args.batch_name}({args.batchNum})_{frame_num:06}.png'
                            blend_factor = ((frame_num % int(turbo_steps)) + 1) / int(turbo_steps)
                            print('turbo skip this frame: skipping clip diffusion steps and saving blended frame')
                            if warp_mode == 'use_image':
                                newWarpedImg = cv2.imread('prevFrameScaled.png')  # this is already updated..
                                oldWarpedImg = cv2.imread('oldFrameScaled.png')
                                blendedImage = cv2.addWeighted(newWarpedImg, blend_factor, oldWarpedImg, 1 - blend_factor, 0.0)
                                cv2.imwrite(f'{batchFolder}/{filename}', blendedImage)
                                next_step_pil.save(f'{img_filepath}')  # save it also as prev_frame to feed next iteration
                            if warp_mode == 'use_latent':
                                newWarpedImg = torch.load('prevFrameScaled_lat.pt')  # this is already updated..
                                oldWarpedImg = torch.load('oldFrameScaled_lat.pt')
                                blendedImage = newWarpedImg * (blend_factor) + oldWarpedImg * (1 - blend_factor)
                                blendedImage = get_image_from_lat(blendedImage).save(f'{batchFolder}/{filename}')
                                torch.save(next_step_pil, f'{img_filepath[:-4]}_lat.pt')

                            if turbo_frame_skips_steps is not None:
                                if warp_mode == 'use_image':
                                    oldWarpedImg = cv2.imread('prevFrameScaled.png')
                                    cv2.imwrite(f'oldFrameScaled.png', oldWarpedImg)  # swap in for blending later
                                print('clip/diff this frame - generate clip diff image')
                                if warp_mode == 'use_latent':
                                    oldWarpedImg = torch.load('prevFrameScaled_lat.pt')
                                    torch.save(
                                        oldWarpedImg,
                                        f'oldFrameScaled_lat.pt',
                                    )  # swap in for blending later
                                skip_steps = math.floor(steps * turbo_frame_skips_steps)
                            else:
                                continue
                        else:
                            # if not a skip frame, will run diffusion and need to blend.
                            if warp_mode == 'use_image':
                                oldWarpedImg = cv2.imread('prevFrameScaled.png')
                                cv2.imwrite(f'oldFrameScaled.png', oldWarpedImg)  # swap in for blending later
                            print('clip/diff this frame - generate clip diff image')
                            if warp_mode == 'use_latent':
                                oldWarpedImg = torch.load('prevFrameScaled_lat.pt')
                                torch.save(
                                    oldWarpedImg,
                                    f'oldFrameScaled_lat.pt',
                                )  # swap in for blending later
                            # oldWarpedImg = cv2.imread('prevFrameScaled.png')
                            # cv2.imwrite(f'oldFrameScaled.png',oldWarpedImg)#swap in for blending later
                            print('clip/diff this frame - generate clip diff image')
                if warp_mode == 'use_image':
                    init_image = 'prevFrameScaled.png'
                else:
                    init_image = 'prevFrameScaled_lat.pt'
                if use_background_mask:
                    if warp_mode == 'use_latent':
                        # pass
                        latent = apply_mask(latent.cpu(), frame_num, background, background_source, invert_mask, warp_mode)  # .save(init_image)

                    if warp_mode == 'use_image':
                        apply_mask(Image.open(init_image), frame_num, background, background_source, invert_mask).save(init_image)
                # init_scale = args.frames_scale
                init_scale = get_scheduled_arg(frame_num, init_scale_schedule)
                # init_latent_scale = args.frames_latent_scale
                init_latent_scale = get_scheduled_arg(frame_num, latent_scale_schedule)

        loss_values = []

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        target_embeds, weights = [], []

        if args.prompts_series is not None and frame_num >= len(args.prompts_series):
            # frame_prompt = args.prompts_series[-1]
            frame_prompt = get_sched_from_json(frame_num, args.prompts_series, blend=False)
        elif args.prompts_series is not None:
            # frame_prompt = args.prompts_series[frame_num]
            frame_prompt = get_sched_from_json(frame_num, args.prompts_series, blend=False)
        else:
            frame_prompt = []

        if VERBOSE: print(args.image_prompts_series)
        if args.image_prompts_series is not None and frame_num >= len(args.image_prompts_series):
            image_prompt = get_sched_from_json(frame_num, args.image_prompts_series, blend=False)
        elif args.image_prompts_series is not None:
            image_prompt = get_sched_from_json(frame_num, args.image_prompts_series, blend=False)
        else:
            image_prompt = []

        init = None

        image_display = Output()
        for i in range(args.n_batches):
            if args.animation_mode == 'None':
                display.clear_output(wait=True)
                batchBar = tqdm(range(args.n_batches), desc="Batches")
                batchBar.n = i
                batchBar.refresh()
            print('')
            display.display(image_display)
            gc.collect()
            torch.cuda.empty_cache()
            steps = int(get_scheduled_arg(frame_num, steps_schedule))
            style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
            skip_steps = int(steps - steps * style_strength)

            if perlin_init:
                init = regen_perlin()

            consistency_mask = None
            if (check_consistency or (model_version == 'v1_inpainting') or ('control_sd15_inpaint' in controlnet_multimodel.keys())) and frame_num > 0:
                frame1_path = f'{videoFramesFolder}/{frame_num:06}.jpg'
                if reverse_cc_order:
                    weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
                else:
                    weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}_12-21_cc.jpg"
                consistency_mask = load_cc(weights_path, blur=consistency_blur, dilate=consistency_dilate)

            if diffusion_model == 'stable_diffusion':
                if VERBOSE: print(args.side_x, args.side_y, init_image)
                # init = Image.open(fetch(init_image)).convert('RGB')

                # init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
                # init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)
                # text_prompt = copy.copy(args.prompts_series[frame_num])
                text_prompt = copy.copy(get_sched_from_json(frame_num, args.prompts_series, blend=False))
                if VERBOSE: print(f'Frame {frame_num} Prompt: {text_prompt}')
                text_prompt = [re.sub('\<(.*?)\>', '', o).strip(' ') for o in text_prompt]  # remove loras from prompt
                used_loras, used_loras_weights = get_loras_weights_for_frame(frame_num, new_prompt_loras)
                if VERBOSE:
                    print('used_loras, used_loras_weights', used_loras, used_loras_weights)
                # used_loras_weights = [o for o in used_loras_weights if o is not None else 0.]
                load_loras(used_loras, used_loras_weights)
                caption = get_caption(frame_num)
                if caption:
                    # print('args.prompt_series',args.prompts_series[frame_num])
                    if '{caption}' in text_prompt[0]:
                        print('Replacing ', '{caption}', 'with ', caption)
                        text_prompt[0] = text_prompt[0].replace('{caption}', caption)
                prompt_patterns = get_sched_from_json(frame_num, prompt_patterns_sched, blend=False)
                if prompt_patterns:
                    for key in prompt_patterns.keys():
                        if key in text_prompt[0]:
                            print('Replacing ', key, 'with ', prompt_patterns[key])
                            text_prompt[0] = text_prompt[0].replace(key, prompt_patterns[key])

                if args.neg_prompts_series is not None:
                    neg_prompt = get_sched_from_json(frame_num, args.neg_prompts_series, blend=False)
                else:
                    neg_prompt = copy.copy(text_prompt)

                if VERBOSE: print(f'Frame {frame_num} neg_prompt: {neg_prompt}')
                if args.rec_prompts_series is not None:
                    rec_prompt = copy.copy(get_sched_from_json(frame_num, args.rec_prompts_series, blend=False))
                    if caption and '{caption}' in rec_prompt[0]:
                        print('Replacing ', '{caption}', 'with ', caption)
                        rec_prompt[0] = rec_prompt[0].replace('{caption}', caption)
                else:
                    rec_prompt = copy.copy(text_prompt)
                if VERBOSE: print(f'Frame {rec_prompt} rec_prompt: {rec_prompt}')

                if VERBOSE:
                    print(neg_prompt, 'neg_prompt')
                    print('init_scale pre sd run', init_scale)
                # init_latent_scale = args.init_latent_scale
                # if frame_num>0:
                #   init_latent_scale = args.frames_latent_scale
                steps = int(get_scheduled_arg(frame_num, steps_schedule))
                init_scale = get_scheduled_arg(frame_num, init_scale_schedule)
                init_latent_scale = get_scheduled_arg(frame_num, latent_scale_schedule)
                style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
                skip_steps = int(steps - steps * style_strength)
                cfg_scale = get_scheduled_arg(frame_num, cfg_scale_schedule)
                image_scale = get_scheduled_arg(frame_num, image_scale_schedule)
                if VERBOSE: printf('skip_steps b4 run_sd: ', skip_steps)

                deflicker_src = {
                    'processed1': f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num - 1:06}.png',
                    'raw1': f'{videoFramesFolder}/{frame_num:06}.jpg',
                    'raw2': f'{videoFramesFolder}/{frame_num + 1:06}.jpg',
                }

                init_grad_img = None
                if init_grad: init_grad_img = f'{videoFramesFolder}/{frame_num + 1:06}.jpg'
                # setup depth source
                if cond_image_src == 'init':
                    cond_image = f'{videoFramesFolder}/{frame_num + 1:06}.jpg'
                if cond_image_src == 'stylized':
                    cond_image = init_image
                if cond_image_src == 'cond_video':
                    cond_image = f'{condVideoFramesFolder}/{frame_num + 1:06}.jpg'

                ref_image = None
                if reference_source == 'init':
                    ref_image = f'{videoFramesFolder}/{frame_num + 1:06}.jpg'
                if reference_source == 'stylized':
                    ref_image = init_image
                if reference_source == 'prev_frame':
                    ref_image = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num - 1:06}.png'
                if reference_source == 'color_video':
                    if os.path.exists(f'{colorVideoFramesFolder}/{frame_num + 1:06}.jpg'):
                        ref_image = f'{colorVideoFramesFolder}/{frame_num + 1:06}.jpg'
                    elif os.path.exists(f'{colorVideoFramesFolder}/{1:06}.jpg'):
                        ref_image = f'{colorVideoFramesFolder}/{1:06}.jpg'
                    else:
                        raise Exception("Reference mode specified with no color video or image. Please specify color video or disable the shuffle model")

                # setup shuffle
                shuffle_source = None
                if 'control_sd15_shuffle' in controlnet_multimodel.keys():
                    if control_sd15_shuffle_source == 'color_video':
                        if os.path.exists(f'{colorVideoFramesFolder}/{frame_num + 1:06}.jpg'):
                            shuffle_source = f'{colorVideoFramesFolder}/{frame_num + 1:06}.jpg'
                        elif os.path.exists(f'{colorVideoFramesFolder}/{1:06}.jpg'):
                            shuffle_source = f'{colorVideoFramesFolder}/{1:06}.jpg'
                        else:
                            raise Exception("Shuffle controlnet specified with no color video or image. Please specify color video or disable the shuffle model")
                    elif control_sd15_shuffle_source == 'init':
                        shuffle_source = init_image
                    elif control_sd15_shuffle_source == 'first_frame':
                        shuffle_source = f'{batchFolder}/{args.batch_name}({args.batchNum})_{0:06}.png'
                    elif control_sd15_shuffle_source == 'prev_frame':
                        shuffle_source = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num - 1:06}.png'
                    if not os.path.exists(shuffle_source):
                        if control_sd15_shuffle_1st_source == 'init':
                            shuffle_source = init_image
                        elif control_sd15_shuffle_1st_source == None:
                            shuffle_source = None
                        elif control_sd15_shuffle_1st_source == 'color_video':
                            if os.path.exists(f'{colorVideoFramesFolder}/{frame_num + 1:06}.jpg'):
                                shuffle_source = f'{colorVideoFramesFolder}/{frame_num + 1:06}.jpg'
                            elif os.path.exists(f'{colorVideoFramesFolder}/{1:06}.jpg'):
                                shuffle_source = f'{colorVideoFramesFolder}/{1:06}.jpg'
                            else:
                                raise Exception("Shuffle controlnet specified with no color video or image. Please specify color video or disable the shuffle model")
                    print('Shuffle source ', shuffle_source)

                # setup temporal source
                if temporalnet_source == 'init':
                    prev_frame = f'{videoFramesFolder}/{frame_num:06}.jpg'
                if temporalnet_source == 'stylized':
                    prev_frame = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num - 1:06}.png'
                if temporalnet_source == 'cond_video':
                    prev_frame = f'{condVideoFramesFolder}/{frame_num:06}.jpg'
                if not os.path.exists(prev_frame):
                    if temporalnet_skip_1st_frame:
                        print('prev_frame not found, replacing 1st videoframe init')
                        prev_frame = None
                    else:
                        prev_frame = f'{videoFramesFolder}/{frame_num + 1:06}.jpg'

                # setup rec noise source
                if rec_source == 'stylized':
                    rec_frame = init_image
                elif rec_source == 'init':
                    rec_frame = f'{videoFramesFolder}/{frame_num + 1:06}.jpg'

                # setup masks for inpainting model
                if model_version == 'v1_inpainting':
                    if inpainting_mask_source == 'consistency_mask':
                        cond_image = consistency_mask
                    if inpainting_mask_source in ['none', None, '', 'None', 'off']:
                        cond_image = None
                    if inpainting_mask_source == 'cond_video': cond_image = f'{condVideoFramesFolder}/{frame_num + 1:06}.jpg'
                    # print('cond_image0',cond_image)

                # setup masks for controlnet inpainting model
                control_inpainting_mask = None
                if 'control_sd15_inpaint' in controlnet_multimodel.keys():
                    if control_sd15_inpaint_mask_source == 'consistency_mask':
                        control_inpainting_mask = consistency_mask
                    if control_sd15_inpaint_mask_source in ['none', None, '', 'None', 'off']:
                        # control_inpainting_mask = None
                        control_inpainting_mask = np.ones((args.side_y, args.side_x, 3))
                    if control_sd15_inpaint_mask_source == 'cond_video':
                        control_inpainting_mask = f'{condVideoFramesFolder}/{frame_num + 1:06}.jpg'
                        control_inpainting_mask = np.array(PIL.Image.open(control_inpainting_mask))
                        # print('cond_image0',cond_image)

                np_alpha = None
                if alpha_masked_diffusion and frame_num > args.start_frame:
                    if VERBOSE: print('Using alpha masked diffusion')
                    print(f'{videoFramesAlpha}/{frame_num + 1:06}.jpg')
                    if videoFramesAlpha == videoFramesFolder or not os.path.exists(f'{videoFramesAlpha}/{frame_num + 1:06}.jpg'):
                        raise Exception(
                            'You have enabled alpha_masked_diffusion without providing an alpha mask source. Please go to mask cell and specify a masked video init or extract a mask from init video.')

                    init_image_alpha = Image.open(f'{videoFramesAlpha}/{frame_num + 1:06}.jpg').resize((args.side_x, args.side_y)).convert('L')
                    np_alpha = np.array(init_image_alpha) / 255.

                sample, latent, depth_img = run_sd(args,
                                                   init_image=init_image,
                                                   skip_timesteps=skip_steps,
                                                   H=args.side_y,
                                                   W=args.side_x,
                                                   text_prompt=text_prompt,
                                                   neg_prompt=neg_prompt,
                                                   steps=steps,
                                                   seed=seed,
                                                   init_scale=init_scale,
                                                   init_latent_scale=init_latent_scale,
                                                   cond_image=cond_image,
                                                   cfg_scale=cfg_scale,
                                                   image_scale=image_scale,
                                                   cond_fn=None,
                                                   init_grad_img=init_grad_img,
                                                   consistency_mask=consistency_mask,
                                                   frame_num=frame_num,
                                                   deflicker_src=deflicker_src,
                                                   prev_frame=prev_frame,
                                                   rec_prompt=rec_prompt,
                                                   rec_frame=rec_frame,
                                                   control_inpainting_mask=control_inpainting_mask,
                                                   shuffle_source=shuffle_source,
                                                   ref_image=ref_image,
                                                   alpha_mask=np_alpha)

                settings_json = save_settings(skip_save=True)
                settings_exif = json2exif(settings_json)

                # depth_img.save(f'{root_dir}/depth_{frame_num}.png')
                filename = f'{args.batch_name}({args.batchNum})_{frame_num:06}.png'
                # if warp_mode == 'use_raw':torch.save(sample,f'{batchFolder}/{filename[:-4]}_raw.pt')
                if warp_mode == 'use_latent':
                    torch.save(latent, f'{batchFolder}/{filename[:-4]}_lat.pt')
                samples = sample * (steps - skip_steps)
                samples = [{"pred_xstart": sample} for sample in samples]
                # for j, sample in enumerate(samples):
                # print(j, sample["pred_xstart"].size)
                # raise Exception
                if VERBOSE: print(sample[0][0].shape)
                image = sample[0][0]
                if do_softcap:
                    image = softcap(image, thresh=softcap_thresh, q=softcap_q)
                image = image.add(1).div(2).clamp(0, 1)
                image = TF.to_pil_image(image)
                if warp_towards_init != 'off' and frame_num != 0:
                    if warp_towards_init == 'init':
                        warp_init_filename = f'{videoFramesFolder}/{frame_num + 1:06}.jpg'
                    else:
                        warp_init_filename = init_image
                    print('warping towards init')
                    init_pil = Image.open(warp_init_filename)
                    image = warp_towards_init_fn(image, init_pil)

                display.clear_output(wait=True)
                fit(image, display_size).save('progress.png', exif=settings_exif)
                display.display(display.Image('progress.png'))

                if mask_result and check_consistency and frame_num > 0:

                    if VERBOSE: print('imitating inpaint')
                    frame1_path = f'{videoFramesFolder}/{frame_num:06}.jpg'
                    weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
                    consistency_mask = load_cc(weights_path, blur=consistency_blur, dilate=consistency_dilate)

                    consistency_mask = cv2.GaussianBlur(consistency_mask, (diffuse_inpaint_mask_blur, diffuse_inpaint_mask_blur), cv2.BORDER_DEFAULT)
                    if diffuse_inpaint_mask_thresh < 1:
                        consistency_mask = np.where(consistency_mask < diffuse_inpaint_mask_thresh, 0, 1.)
                    # if dither:
                    #   consistency_mask = Dither.dither(consistency_mask, 'simple2D', resize=True)

                    # consistency_mask = torchvision.transforms.functional.resize(consistency_mask, image.size)
                    if warp_mode == 'use_image':
                        consistency_mask = cv2.GaussianBlur(consistency_mask, (3, 3), cv2.BORDER_DEFAULT)
                        init_img_prev = Image.open(init_image)
                        if VERBOSE: print(init_img_prev.size, consistency_mask.shape, image.size)
                        cc_sz = consistency_mask.shape[1], consistency_mask.shape[0]
                        image_masked = np.array(image) * (1 - consistency_mask) + np.array(init_img_prev) * (consistency_mask)

                        # image_masked = np.array(image.resize(cc_sz, warp_interp))*(1-consistency_mask) + np.array(init_img_prev.resize(cc_sz, warp_interp))*(consistency_mask)
                        image_masked = Image.fromarray(image_masked.round().astype('uint8'))
                        # image = image_masked.resize(image.size, warp_interp)
                        image = image_masked
                    if warp_mode == 'use_latent':
                        if invert_mask: consistency_mask = 1 - consistency_mask
                        init_lat_prev = torch.load('prevFrameScaled_lat.pt')
                        sample_masked = sd_model.decode_first_stage(latent.cuda())[0]
                        image_prev = TF.to_pil_image(sample_masked.add(1).div(2).clamp(0, 1))

                        cc_small = consistency_mask[::8, ::8, 0]
                        latent = latent.cpu() * (1 - cc_small) + init_lat_prev * cc_small
                        torch.save(latent, 'prevFrameScaled_lat.pt')

                        # image_prev = Image.open(f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-1:06}.png')
                        torch.save(latent, 'prevFrame_lat.pt')
                        # cc_sz = consistency_mask.shape[1], consistency_mask.shape[0]
                        # image_prev = Image.open('prevFrameScaled.png')
                        image_masked = np.array(image) * (1 - consistency_mask) + np.array(image_prev) * (consistency_mask)

                        # # image_masked = np.array(image.resize(cc_sz, warp_interp))*(1-consistency_mask) + np.array(init_img_prev.resize(cc_sz, warp_interp))*(consistency_mask)
                        image_masked = Image.fromarray(image_masked.round().astype('uint8'))
                        # image = image_masked.resize(image.size, warp_interp)
                        image = image_masked

                if (frame_num > args.start_frame) or ('color_video' in normalize_latent):
                    global first_latent
                    global first_latent_source

                    if 'frame' in normalize_latent:

                        def img2latent(img_path):
                            frame2 = Image.open(img_path)
                            frame2pil = frame2.convert('RGB').resize(image.size, warp_interp)
                            frame2pil = np.array(frame2pil)
                            frame2pil = (frame2pil / 255.)[None, ...].transpose(0, 3, 1, 2)
                            frame2pil = 2 * torch.from_numpy(frame2pil).float().cuda() - 1.
                            frame2pil = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(frame2pil))
                            return frame2pil

                        try:
                            if VERBOSE: print('Matching latent to:')
                            filename = get_frame_from_color_mode(normalize_latent, normalize_latent_offset, frame_num)
                            match_latent = img2latent(filename)
                            first_latent = match_latent
                            first_latent_source = filename
                            # print(first_latent_source, first_latent)
                        except:
                            if VERBOSE: print(traceback.format_exc())
                            print(f'Frame with offset/position {normalize_latent_offset} not found')
                            if 'init' in normalize_latent:
                                try:
                                    filename = f'{videoFramesFolder}/{0:06}.jpg'
                                    match_latent = img2latent(filename)
                                    first_latent = match_latent
                                    first_latent_source = filename
                                except:
                                    pass
                            print(f'Color matching the 1st frame.')

                    if colormatch_frame != 'off' and colormatch_after:
                        if not turbo_mode & (frame_num % int(turbo_steps) != 0) or colormatch_turbo:
                            try:
                                print('Matching color to:')
                                filename = get_frame_from_color_mode(colormatch_frame, colormatch_offset, frame_num)
                                match_frame = Image.open(filename)
                                first_frame = match_frame
                                first_frame_source = filename

                            except:
                                print(f'Frame with offset/position {colormatch_offset} not found')
                                if 'init' in colormatch_frame:
                                    try:
                                        filename = f'{videoFramesFolder}/{1:06}.jpg'
                                        match_frame = Image.open(filename)
                                        first_frame = match_frame
                                        first_frame_source = filename
                                    except:
                                        pass
                                print(f'Color matching the 1st frame.')
                            print('Colormatch source - ', first_frame_source)
                            image = Image.fromarray(match_color_var(first_frame, image, opacity=color_match_frame_str, f=colormatch_method_fn, regrain=colormatch_regrain))

                if frame_num == args.start_frame:
                    settings_json = save_settings()
                if args.animation_mode != "None":
                    # sys.exit(os.getcwd(), 'cwd')
                    if warp_mode == 'use_image':
                        image.save('prevFrame.png', exif=settings_exif)
                    else:
                        torch.save(latent, 'prevFrame_lat.pt')
                filename = f'{args.batch_name}({args.batchNum})_{frame_num:06}.png'
                image.save(f'{batchFolder}/{filename}', exif=settings_exif)
                # np.save(latent, f'{batchFolder}/{filename[:-4]}.npy')
                if args.animation_mode == 'Video Input':
                    # If turbo, save a blended image
                    if turbo_mode and frame_num > args.start_frame:
                        # Mix new image with prevFrameScaled
                        blend_factor = (1) / int(turbo_steps)
                        if warp_mode == 'use_image':
                            newFrame = cv2.imread('prevFrame.png')  # This is already updated..
                            prev_frame_warped = cv2.imread('prevFrameScaled.png')
                            blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped, (1 - blend_factor), 0.0)
                            cv2.imwrite(f'{batchFolder}/{filename}', blendedImage)
                        if warp_mode == 'use_latent':
                            newFrame = torch.load('prevFrame_lat.pt').cuda()
                            prev_frame_warped = torch.load('prevFrameScaled_lat.pt').cuda()
                            blendedImage = newFrame * (blend_factor) + prev_frame_warped * (1 - blend_factor)
                            blendedImage = get_image_from_lat(blendedImage)
                            blendedImage.save(f'{batchFolder}/{filename}', exif=settings_exif)

                else:
                    image.save(f'{batchFolder}/{filename}', exif=settings_exif)
                    image.save('prevFrameScaled.png', exif=settings_exif)

            plt.plot(np.array(loss_values), 'r')
    batchBar.close()
