// Harness executes each native call and carries its result back to the model.
// Python owns scopes, durable receipts, async services and permissions.
export const inject = ['tools', 'attachments', 'agents'];

export async function apply(ctx) {
  // 0.1.5rc1 SDK server calls create() for already persisted ids. Use its
  // public resume API for this exact upstream compatibility defect; the
  // native session store validates replay, orphaned calls and closed turns.
  const registry = ctx.agents;
  const create = registry.create.bind(registry);
  registry.create = async options => {
    try { return await create(options); }
    catch (error) {
      if (registry.get(options.sessionId) || error?.message !== `session "${options.sessionId}" already exists`) throw error;
      return registry.resume({
        resumeSessionId: options.sessionId, agentOptions: options.agentOptions,
        ...(options.setup ? { setup: options.setup } : {}),
      });
    }
  };
  const hasImage = value => value && typeof value === 'object' &&
    (value.type === 'image' || Object.values(value).some(hasImage));
  let receivedImage = process.env.XIAOMACHI_HAS_IMAGES === '1';
  ctx.on('agent/request', async ({ agent }, next) => {
    const config = await next();
    const images = receivedImage || agent.session.deriveMessages().some(hasImage);
    return { ...config, model: images ? process.env.XIAOMACHI_VISION_MODEL : process.env.XIAOMACHI_PRIMARY_MODEL };
  });
  const base = process.env.XIAOMACHI_BRIDGE_URL;
  const headers = { Authorization: `Bearer ${process.env.XIAOMACHI_BRIDGE_TOKEN}` };
  const response = await fetch(`${base}/tools`, { headers });
  if (!response.ok) throw new Error('Python tool registry unavailable');
  for (const schema of await response.json()) {
    ctx.tools.register({
      ...schema,
      output: {
        schema: { type: 'object' },
        render: (_args, value) => [
          { type: 'text', text: JSON.stringify(value.result) },
          ...value.attachments.map(attachment => ({ type: 'image', attachment })),
        ],
      },
      async execute(args, execution) {
        const reply = await fetch(`${base}/tools/call`, {
          method: 'POST', signal: execution.signal,
          headers: { ...headers, 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: schema.name, arguments: args, call_id: execution.callId }),
        });
        if (!reply.ok) throw new Error('Python tool call interrupted');
        const { images = [], ...result } = await reply.json();
        const attachments = [];
        for (const image of images) {
          receivedImage = true;
          execution.signal.throwIfAborted();
          attachments.push(await ctx.attachments.saveImage({
            data: Buffer.from(image.data, 'base64'), mediaType: image.mediaType,
          }));
        }
        return { result, attachments };
      },
    });
  }
}
