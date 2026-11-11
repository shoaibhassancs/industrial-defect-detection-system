import { describe, expect, it } from "vitest";
import { jpegFile, pngFile, textFile } from "../../test/fixtures";
import { checkFile, MAX_UPLOAD_MB } from "./validateFile";

describe("checkFile", () => {
  it("accepts a PNG and a JPEG by their first bytes", async () => {
    expect(await checkFile(pngFile())).toBeNull();
    expect(await checkFile(jpegFile())).toBeNull();
  });

  it("refuses a text file even when it is named .png and claims image/png", async () => {
    expect((await checkFile(textFile()))?.code).toBe("NOT_JPEG_OR_PNG");
  });

  it("refuses an empty file", async () => {
    expect((await checkFile(new File([], "empty.png", { type: "image/png" })))?.code).toBe("EMPTY");
  });

  it("refuses a file over the limit without reading all of it", async () => {
    const big = pngFile("big.png", MAX_UPLOAD_MB * 1024 * 1024 + 1);
    expect((await checkFile(big))?.code).toBe("TOO_LARGE");
  });

  it("accepts a file exactly at the limit", async () => {
    expect(await checkFile(pngFile("edge.png", MAX_UPLOAD_MB * 1024 * 1024))).toBeNull();
  });
});
