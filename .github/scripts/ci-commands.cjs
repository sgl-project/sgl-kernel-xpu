const workflows = ["pr-test-xpu.yml", "lint.yml"];
const failedConclusions = new Set([
  "failure", "timed_out", "cancelled", "action_required", "startup_failure", "stale",
]);

module.exports = async function handleCommand({ github, context, core }) {
  const { issue, comment, repository } = context.payload;
  const command = comment.body.trim();
  if (!issue.pull_request || comment.user.type === "Bot" ||
      !["/rerun-ci", "/rerun-failed-ci"].includes(command)) {
    return;
  }

  const repo = context.repo;
  const reply = (body) => github.rest.issues.createComment({
    ...repo, issue_number: issue.number, body,
  });
  const { data: permission } = await github.rest.repos.getCollaboratorPermissionLevel({
    ...repo, username: comment.user.login,
  });
  if (!["admin", "maintain", "write"].includes(permission.permission)) {
    await reply("CI commands require write, maintain, or admin repository permission.");
    return;
  }

  const { data: pr } = await github.rest.pulls.get({ ...repo, pull_number: issue.number });
  if (pr.state !== "open" || pr.base.ref !== repository.default_branch) {
    await reply("CI commands require an open PR targeting the default branch.");
    return;
  }
  if (!pr.labels.some((label) => label.name === "run-ci")) {
    await reply("Add the `run-ci` label before requesting CI.");
    return;
  }

  const messages = [];
  for (const workflow of workflows) {
    const runs = await github.paginate(github.rest.actions.listWorkflowRuns, {
      ...repo, workflow_id: workflow, event: "pull_request", head_sha: pr.head.sha,
      per_page: 100,
    });
    const candidates = runs.filter((run) => run.head_sha === pr.head.sha &&
      (run.pull_requests.some((pull) => pull.number === pr.number) ||
       (run.head_branch === pr.head.ref &&
        run.head_repository.full_name === pr.head.repo.full_name)));
    if (workflow === "pr-test-xpu.yml") {
      const dispatched = await github.paginate(github.rest.actions.listWorkflowRuns, {
        ...repo, workflow_id: workflow, event: "workflow_dispatch",
        created: `>=${pr.created_at}`, per_page: 100,
      });
      const title = `PR Test (XPU) #${pr.number} @ ${pr.head.sha}`;
      candidates.push(...dispatched.filter((run) => run.display_title === title));
    }
    const latest = candidates.sort((left, right) => right.id - left.id)[0];
    if (latest && latest.status !== "completed") {
      messages.push(`${workflow}: already active: ${latest.html_url}`);
      continue;
    }

    if (command === "/rerun-failed-ci") {
      if (latest && failedConclusions.has(latest.conclusion)) {
        await github.rest.actions.reRunWorkflowFailedJobs({ ...repo, run_id: latest.id });
        messages.push(`${workflow}: requested failed-job rerun: ${latest.html_url}`);
      } else {
        messages.push(`${workflow}: no failed run for the current PR head.`);
      }
    } else if (workflow === "pr-test-xpu.yml") {
      await github.rest.actions.createWorkflowDispatch({
        ...repo, workflow_id: workflow, ref: repository.default_branch,
        inputs: { pr_number: String(pr.number), head_sha: pr.head.sha },
      });
      messages.push(`${workflow}: dispatched for PR #${pr.number} at \`${pr.head.sha}\`.`);
    } else if (latest && latest.conclusion !== "skipped") {
      await github.rest.actions.reRunWorkflow({ ...repo, run_id: latest.id });
      messages.push(`${workflow}: requested full rerun: ${latest.html_url}`);
    } else {
      messages.push(`${workflow}: no eligible run for the current PR head.`);
    }
  }
  core.info(messages.join("\n"));
  await reply(messages.join("\n\n"));
};

module.exports.resolvePr = async function resolvePr({ github, context, core }) {
  const inputs = context.payload.inputs || {};
  const number = context.payload.pull_request?.number || inputs.pr_number;
  if (!/^[1-9]\d*$/.test(String(number))) {
    throw new Error("A positive PR number is required.");
  }
  const { data: pr } = await github.rest.pulls.get({
    ...context.repo, pull_number: Number(number),
  });
  if (pr.state !== "open" || pr.base.ref !== context.payload.repository.default_branch ||
      !pr.labels.some((label) => label.name === "run-ci")) {
    throw new Error("The PR must be open, target the default branch, and have run-ci.");
  }
  const expected = context.payload.pull_request?.head.sha || inputs.head_sha;
  if (expected !== pr.head.sha) {
    throw new Error("The PR head changed. Request CI again for the current head.");
  }
  core.setOutput("number", String(pr.number));
  core.setOutput("sha", pr.head.sha);
};
