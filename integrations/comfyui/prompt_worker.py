# Derived from ComfyUI prompt_worker; see NOTICE and LICENSE.
def prompt_worker(q, server_instance):
    current_time: float = 0.0
    cache_ram = 0
    cache_ram_inactive = 0
    if not args.cache_classic and not args.cache_none and args.cache_lru <= 0:
        cache_ram = min(10.0, max(2.0, comfy.model_management.total_ram * 0.10 / 1024.0))
        cache_ram_inactive = min(128.0, comfy.model_management.total_ram / 1024.0)
        if len(args.cache_ram) > 0:
            cache_ram = args.cache_ram[0]
        if len(args.cache_ram) > 1:
            cache_ram_inactive = args.cache_ram[1]

    cache_type = execution.CacheType.RAM_PRESSURE
    if args.cache_classic:
        cache_type = execution.CacheType.CLASSIC
    elif args.cache_lru > 0:
        cache_type = execution.CacheType.LRU
    elif args.cache_none:
        cache_type = execution.CacheType.NONE

    e = execution.PromptExecutor(server_instance, cache_type=cache_type, cache_args={ "lru" : args.cache_lru, "ram" : cache_ram, "ram_inactive" : cache_ram_inactive } )
    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0

    while True:
        timeout = 1000.0
        if need_gc:
            timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

        queue_item = q.get(timeout=timeout)
        if queue_item is not None:
            item, item_id = queue_item
            execution_start_time = time.perf_counter()
            prompt_id = item[1]
            server_instance.last_prompt_id = prompt_id

            sensitive = item[5]
            extra_data = item[3].copy()
            for k in sensitive:
                extra_data[k] = sensitive[k]

            asset_seeder.pause()
            release_between = extra_data.get("release_memory_between_prompts") is True
            cleanup_state = server_instance.queue_cleanup_state
            pre_cleanup = None
            if release_between:
                cleanup_state.update(prompt_id=prompt_id, boundary="before")
                try:
                    pre_cleanup = release_prompt_memory(e, comfy.model_management, comfy.memory_management, cleanup_state)
                except Exception as exc:
                    logging.exception("Memory cleanup failed before prompt; restarting with queue preserved")
                    cleanup_state.update(phase='restart_required', error=str(exc))
                    restart_with_queue_checkpoint(q, server_instance, item_id, None, None)
                cleanup_state["phase"] = "executing"
            cleanup_state["execution_started_at"] = time.time()
            e.execute(item[2], prompt_id, extra_data, item[4])

            # reset() clears success and messages; preserve the real execution
            # result before releasing caches, including failed prompts.
            execution_success = e.success
            execution_messages = list(e.status_messages)
            history_result = e.history_result
            if release_between:
                cleanup_state.update(prompt_id=prompt_id, boundary="after")
                try:
                    post_cleanup = release_prompt_memory(e, comfy.model_management, comfy.memory_management, cleanup_state)
                except Exception as exc:
                    logging.exception("Memory cleanup failed after prompt; restarting with results preserved")
                    cleanup_state.update(phase='restart_required', error=str(exc))
                    history_result['queue_memory_cleanup'] = {
                        'before': pre_cleanup, 'after': {**cleanup_state, 'status': 'restart_checkpoint'}}
                    restart_with_queue_checkpoint(q, server_instance, item_id, history_result,
                        {'status_str': 'success' if execution_success else 'error',
                         'completed': execution_success, 'messages': execution_messages})
                history_result["queue_memory_cleanup"] = {"before": pre_cleanup, "after": post_cleanup}

            need_gc = True

            remove_sensitive = lambda prompt: prompt[:5] + prompt[6:]
            q.task_done(item_id,
                        history_result,
                        status=execution.PromptQueue.ExecutionStatus(
                            status_str='success' if execution_success else 'error',
                            completed=execution_success,
                            messages=execution_messages), process_item=remove_sensitive)
            if server_instance.client_id is not None:
                server_instance.send_sync("executing", {"node": None, "prompt_id": prompt_id}, server_instance.client_id)

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time

            # Log Time in a more readable way after 10 minutes
            if execution_time > 600:
                execution_time = time.strftime("%H:%M:%S", time.gmtime(execution_time))
                logging.info(f"Prompt executed in {execution_time}", extra={'color': 'green'})
            else:
                logging.info("Prompt executed in {:.2f} seconds".format(execution_time), extra={'color': 'green'})

            if not asset_seeder.is_disabled():
                paths = _collect_output_absolute_paths(history_result)
                register_output_files(paths, job_id=prompt_id)

        flags = q.get_flags()
        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):
            comfy.model_management.unload_all_models()
            need_gc = True
            last_gc_collect = 0

        if free_memory:
            e.reset()
            need_gc = True
            last_gc_collect = 0

        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:
                gc.collect()
                comfy.model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False
                hook_breaker_ac10a0.restore_functions()

                if not asset_seeder.is_disabled():
                    asset_seeder.enqueue_enrich(roots=("output",), compute_hashes=args.enable_asset_hashing)
                asset_seeder.resume()
