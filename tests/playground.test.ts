import { describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";

const root = resolve(import.meta.dir, "..");
const playground = resolve(root, "playground");

type PackageJson = {
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
};

describe("playground deployment", () => {
  test("ships the complete browser module dependency tree", async () => {
    const build = Bun.spawn(["bun", "run", "playground/build-pages.ts"], {
      cwd: root,
      stdout: "pipe",
      stderr: "pipe",
    });

    expect(await build.exited).toBe(0);

    const missingImports: string[] = [];
    const files = new Bun.Glob("lib/**/*.js").scan({
      cwd: resolve(playground, "dist"),
      absolute: true,
    });

    for await (const file of files) {
      const source = await Bun.file(file).text();
      const imports = source.matchAll(/(?:from|import)\s*\(?["'](\.{1,2}\/[^"']+)["']/g);

      for (const match of imports) {
        const target = resolve(dirname(file), match[1]);
        if (!existsSync(target)) missingImports.push(`${file} -> ${match[1]}`);
      }
    }

    expect(missingImports).toEqual([]);
  });

  test("ships every sample the page offers", async () => {
    const html = await Bun.file(resolve(playground, "index.html")).text();
    const offered = [...html.matchAll(/data-sample="([^"]+)"/g)].map((m) => m[1] ?? "");
    expect(offered.length).toBeGreaterThan(0);
    for (const name of offered) {
      expect(existsSync(resolve(playground, "dist/samples", name))).toBe(true);
    }
  });

  test("the page is ASCII and pins one library version", async () => {
    const html = await Bun.file(resolve(playground, "index.html")).text();
    // Source files stay ASCII: no emoji, dashes or arrows.
    expect([...html].filter((ch) => ch.charCodeAt(0) > 127)).toEqual([]);

    const pkg: PackageJson & { version?: string } = await Bun.file(
      resolve(root, "package.json")
    ).json();
    const pinned = new Set(
      [...html.matchAll(/ppu-indra-ocr@(\d+\.\d+\.\d+)/g)].map((m) => m[1] ?? "")
    );
    expect([...pinned]).toEqual([pkg.version ?? ""]);
  });

  test("the page keeps the controls the playground drives", async () => {
    const html = await Bun.file(resolve(playground, "index.html")).text();
    for (const id of [
      "file-input",
      "btn-process",
      "cfg-engine",
      "cfg-math",
      "cfg-furniture",
      "cfg-adaptive",
      "result-canvas",
      "overlay-canvas",
      "tab-markdown",
      "tab-lines",
      "tab-blocks",
      "tab-json",
      "license-bar",
    ]) {
      expect(html).toContain(`id="${id}"`);
    }
    expect(html).toContain('<meta name="referrer" content="no-referrer" />');
    expect(html).toContain("Not endorsed by Datalab");
  });

  test("the import map pins every package the browser code imports, at the tested version", async () => {
    const html = await Bun.file(resolve(playground, "index.html")).text();
    const map = html.match(/<script type="importmap">([\s\S]*?)<\/script>/)?.[1] ?? "{}";
    const imports: Record<string, string> = JSON.parse(map).imports ?? {};

    const pkg: PackageJson = await Bun.file(resolve(root, "package.json")).json();
    const versions = { ...pkg.dependencies, ...pkg.devDependencies };
    const exact = (name: string): string => (versions[name] ?? "?").replace(/^[\^~]/, "");

    const bare = new Set<string>();
    const sources = new Bun.Glob("src/{web,pdf,core,indra}/**/*.ts").scan({ cwd: root });
    for await (const file of sources) {
      const source = await Bun.file(resolve(root, file)).text();
      // Static value imports only: type imports are erased from the build.
      for (const m of source.matchAll(/^import (?!type )[^;]*? from "([^".][^"]*)";/gm)) {
        if (m[1]) bare.add(m[1]);
      }
    }
    expect(bare.size).toBeGreaterThan(0);

    for (const name of bare) {
      const parts = name.split("/");
      const pkgName = name.startsWith("@") ? parts.slice(0, 2).join("/") : (parts[0] ?? name);
      expect(imports[name] ?? `missing: ${name}`).toContain(`${pkgName}@${exact(pkgName)}/`);
    }
  });
});
