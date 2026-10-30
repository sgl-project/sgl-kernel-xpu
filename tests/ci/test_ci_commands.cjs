const assert = require("node:assert/strict");
const test = require("node:test");
const handleCommand = require("../../.github/scripts/ci-commands.cjs");

function harness(overrides = {}) {
  const calls = [];
  const pr = {
    number: 532, state: "open", base: { ref: "main" },
    labels: [{ name: "run-ci" }], created_at: "2026-10-01T00:00:00Z",
    head: { sha: "current", ref: "feature", repo: { full_name: "author/kernel" } },
    ...overrides.pr,
  };
  const record = (method) => async (args) => { calls.push({ method, ...args }); };
  const github = {
    rest: {
      repos: { getCollaboratorPermissionLevel: async () => ({
        data: { permission: overrides.permission || "write" },
      }) },
      pulls: { get: async () => ({ data: pr }) },
      issues: { createComment: record("comment") },
      actions: {
        listWorkflowRuns: "list",
        createWorkflowDispatch: record("dispatch"),
        reRunWorkflow: record("rerun"),
        reRunWorkflowFailedJobs: record("failed"),
      },
    },
    paginate: async (_, args) => (overrides.runs || []).filter((run) =>
      run.workflow === args.workflow_id && run.event === args.event),
  };
  const context = {
    repo: { owner: "sgl-project", repo: "sgl-kernel-xpu" },
    payload: {
      repository: { default_branch: "main" },
      issue: { number: 532, pull_request: {} },
      comment: { body: overrides.command || "/rerun-ci",
        user: { login: "maintainer", type: "User" } },
    },
  };
  return { calls, github, context, core: { info: () => {} } };
}

function run(overrides = {}) {
  return {
    id: 100, event: "pull_request", workflow: "pr-test-xpu.yml",
    status: "completed", conclusion: "failure", head_sha: "current",
    pull_requests: [{ number: 532 }], html_url: "https://example.test/run/100",
    ...overrides,
  };
}

test("full rerun dispatches current XPU head and reruns lint", async () => {
  const fixture = harness({ runs: [run({ workflow: "lint.yml" })] });
  await handleCommand(fixture);
  assert.deepEqual(fixture.calls.filter((call) => call.method === "dispatch")[0].inputs,
    { pr_number: "532", head_sha: "current" });
  assert.equal(fixture.calls.filter((call) => call.method === "rerun").length, 1);
});

test("skipped label snapshot starts a new XPU run, not an old rerun", async () => {
  const fixture = harness({ runs: [run({ conclusion: "skipped" })] });
  await handleCommand(fixture);
  assert.equal(fixture.calls.filter((call) => call.method === "dispatch").length, 1);
  assert.equal(fixture.calls.filter((call) => call.method === "rerun").length, 0);
});

for (const permission of ["read", "triage"]) {
  test(`rejects ${permission} permission`, async () => {
    const fixture = harness({ permission });
    await handleCommand(fixture);
    assert.deepEqual(fixture.calls.map((call) => call.method), ["comment"]);
  });
}

for (const pr of [{ labels: [] }, { state: "closed" }, { base: { ref: "release" } }]) {
  test(`rejects ineligible PR ${JSON.stringify(pr)}`, async () => {
    const fixture = harness({ pr });
    await handleCommand(fixture);
    assert.deepEqual(fixture.calls.map((call) => call.method), ["comment"]);
  });
}

test("failed-only ignores old heads and successful runs", async () => {
  const fixture = harness({ command: "/rerun-failed-ci", runs: [
    run({ head_sha: "old" }), run({ id: 101, conclusion: "success" }),
  ] });
  await handleCommand(fixture);
  assert.deepEqual(fixture.calls.map((call) => call.method), ["comment"]);
});

test("failed-only reruns failed jobs from the latest dispatch", async () => {
  const fixture = harness({ command: "/rerun-failed-ci", runs: [run(), run({
    id: 101, event: "workflow_dispatch", display_title: "PR Test (XPU) #532 @ current",
  })] });
  await handleCommand(fixture);
  assert.equal(fixture.calls.find((call) => call.method === "failed").run_id, 101);
});

test("active run prevents duplicate dispatch", async () => {
  const fixture = harness({ runs: [run({ status: "queued", conclusion: null })] });
  await handleCommand(fixture);
  assert.deepEqual(fixture.calls.map((call) => call.method), ["comment"]);
});

test("does not interpret arguments or injected text as commands", async () => {
  const fixture = harness({ command: "/rerun-ci\nconsole.log('untrusted')" });
  await handleCommand(fixture);
  assert.equal(fixture.calls.length, 0);
});

test("dispatch resolver publishes the current PR head", async () => {
  const fixture = harness();
  const outputs = {};
  fixture.context.payload.inputs = { pr_number: "532", head_sha: "current" };
  fixture.core.setOutput = (key, value) => { outputs[key] = value; };
  await handleCommand.resolvePr(fixture);
  assert.deepEqual(outputs, { number: "532", sha: "current" });
});

test("dispatch resolver rejects a head changed after the command", async () => {
  const fixture = harness();
  fixture.context.payload.inputs = { pr_number: "532", head_sha: "old" };
  await assert.rejects(handleCommand.resolvePr(fixture), /head changed/);
});

test("dispatch resolver rejects removed labels", async () => {
  const fixture = harness({ pr: { labels: [] } });
  fixture.context.payload.inputs = { pr_number: "532", head_sha: "current" };
  await assert.rejects(handleCommand.resolvePr(fixture), /have run-ci/);
});

test("dispatch resolver rejects malformed PR numbers", async () => {
  const fixture = harness();
  fixture.context.payload.inputs = { pr_number: "532; echo unsafe", head_sha: "current" };
  await assert.rejects(handleCommand.resolvePr(fixture), /positive PR number/);
});

test("resolver handles normal PR events", async () => {
  const fixture = harness();
  const outputs = {};
  fixture.context.payload.pull_request = { number: 532, head: { sha: "current" } };
  fixture.core.setOutput = (key, value) => { outputs[key] = value; };
  await handleCommand.resolvePr(fixture);
  assert.equal(outputs.sha, "current");
});

test("ignores bot comments and non-PR issues", async () => {
  for (const kind of ["bot", "issue"]) {
    const fixture = harness();
    if (kind === "bot") {
      fixture.context.payload.comment.user.type = "Bot";
    } else {
      delete fixture.context.payload.issue.pull_request;
    }
    await handleCommand(fixture);
    assert.equal(fixture.calls.length, 0);
  }
});

test("failed-only never reruns another PR sharing the commit", async () => {
  const fixture = harness({ command: "/rerun-failed-ci", runs: [run({
    pull_requests: [{ number: 999 }], head_branch: "other",
    head_repository: { full_name: "other/kernel" },
  }), run({
    id: 101, event: "workflow_dispatch", display_title: "PR Test (XPU) #999 @ current",
  })] });
  await handleCommand(fixture);
  assert.deepEqual(fixture.calls.map((call) => call.method), ["comment"]);
});

test("matches a fork run even when GitHub omits pull_requests", async () => {
  const fixture = harness({ command: "/rerun-failed-ci", runs: [run({
    pull_requests: [], head_branch: "feature",
    head_repository: { full_name: "author/kernel" },
  })] });
  await handleCommand(fixture);
  assert.equal(fixture.calls.find((call) => call.method === "failed").run_id, 100);
});

test("skipped runs are not failed-only rerun candidates", async () => {
  const fixture = harness({ command: "/rerun-failed-ci", runs: [run({ conclusion: "skipped" })] });
  await handleCommand(fixture);
  assert.deepEqual(fixture.calls.map((call) => call.method), ["comment"]);
});
