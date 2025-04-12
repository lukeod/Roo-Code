import path from "path"
import { Cline } from "../Cline"
import { ClineSayTool } from "../../shared/ExtensionMessage"
import { ToolUse } from "../assistant-message"
import { formatResponse } from "../prompts/responses"
import { t } from "../../i18n"
import { AskApproval, HandleError, PushToolResult, RemoveClosingTag } from "./types"
import { RecordSource } from "../context-tracking/FileContextTrackerTypes"
import { isPathOutsideWorkspace } from "../../utils/pathUtils"
import { getReadablePath } from "../../utils/path"
import { countFileLines } from "../../integrations/misc/line-counter"
import { readLines } from "../../integrations/misc/read-lines"
import { extractTextFromFile, addLineNumbers } from "../../integrations/misc/extract-text"
import { parseSourceCodeDefinitionsForFile } from "../../services/tree-sitter"
import { isBinaryFile } from "isbinaryfile"

// Track file read operations
const fileReadStats = {
	totalReads: 0,
	readsByModel: new Map<string, number>(),
	readsByFile: new Map<string, number>(),
	errors: [] as Array<{ time: number; file: string; error: string }>,
	lastReadTime: 0,
	consecutiveReads: 0,
	consecutiveReadsWindow: 60000, // 60 seconds window
}

export async function readFileTool(
	cline: Cline,
	block: ToolUse,
	askApproval: AskApproval,
	handleError: HandleError,
	pushToolResult: PushToolResult,
	removeClosingTag: RemoveClosingTag,
) {
	const startTime = Date.now()
	const currentModelId = cline.api.getModel().id

	// Update global stats
	fileReadStats.totalReads++
	fileReadStats.readsByModel.set(currentModelId, (fileReadStats.readsByModel.get(currentModelId) || 0) + 1)

	// Track consecutive reads in a time window
	const timeSinceLastRead = startTime - fileReadStats.lastReadTime
	if (timeSinceLastRead < fileReadStats.consecutiveReadsWindow) {
		fileReadStats.consecutiveReads++
	} else {
		fileReadStats.consecutiveReads = 1
	}
	fileReadStats.lastReadTime = startTime

	console.log(
		`[readFileTool] Starting file read operation #${fileReadStats.totalReads} | Model: ${currentModelId} | Consecutive reads: ${fileReadStats.consecutiveReads}`,
	)

	const relPath: string | undefined = block.params.path
	const startLineStr: string | undefined = block.params.start_line
	const endLineStr: string | undefined = block.params.end_line

	// Get the full path and determine if it's outside the workspace
	const fullPath = relPath ? path.resolve(cline.cwd, removeClosingTag("path", relPath)) : ""
	const isOutsideWorkspace = isPathOutsideWorkspace(fullPath)

	if (relPath) {
		fileReadStats.readsByFile.set(relPath, (fileReadStats.readsByFile.get(relPath) || 0) + 1)
		console.log(
			`[readFileTool] Reading file: ${relPath} | Full path: ${fullPath} | Read count: ${fileReadStats.readsByFile.get(relPath)}`,
		)
	}

	const sharedMessageProps: ClineSayTool = {
		tool: "readFile",
		path: getReadablePath(cline.cwd, removeClosingTag("path", relPath)),
		isOutsideWorkspace,
	}
	try {
		if (block.partial) {
			console.log(`[readFileTool] Processing partial block`)
			const partialMessage = JSON.stringify({
				...sharedMessageProps,
				content: undefined,
			} satisfies ClineSayTool)
			await cline.ask("tool", partialMessage, block.partial).catch((err) => {
				console.error(`[readFileTool] Error in partial block:`, err)
			})
			return
		} else {
			if (!relPath) {
				cline.consecutiveMistakeCount++
				const errorMsg = await cline.sayAndCreateMissingParamError("read_file", "path")
				pushToolResult(`<file><path></path><error>${errorMsg}</error></file>`)
				return
			}

			// Check if we're doing a line range read
			let isRangeRead = false
			let startLine: number | undefined = undefined
			let endLine: number | undefined = undefined

			// Check if we have either range parameter
			if (startLineStr || endLineStr) {
				isRangeRead = true
			}

			// Parse start_line if provided
			if (startLineStr) {
				startLine = parseInt(startLineStr)
				if (isNaN(startLine)) {
					// Invalid start_line
					cline.consecutiveMistakeCount++
					await cline.say("error", `Failed to parse start_line: ${startLineStr}`)
					pushToolResult(`<file><path>${relPath}</path><error>Invalid start_line value</error></file>`)
					return
				}
				startLine -= 1 // Convert to 0-based index
			}

			// Parse end_line if provided
			if (endLineStr) {
				endLine = parseInt(endLineStr)

				if (isNaN(endLine)) {
					// Invalid end_line
					cline.consecutiveMistakeCount++
					await cline.say("error", `Failed to parse end_line: ${endLineStr}`)
					pushToolResult(`<file><path>${relPath}</path><error>Invalid end_line value</error></file>`)
					return
				}

				// Convert to 0-based index
				endLine -= 1
			}

			const accessAllowed = cline.rooIgnoreController?.validateAccess(relPath)
			if (!accessAllowed) {
				console.log(`[readFileTool] Access denied by rooIgnoreController for ${relPath}`)
				await cline.say("rooignore_error", relPath)
				const errorMsg = formatResponse.rooIgnoreError(relPath)
				pushToolResult(`<file><path>${relPath}</path><error>${errorMsg}</error></file>`)
				return
			}

			const { maxReadFileLine = 500 } = (await cline.providerRef.deref()?.getState()) ?? {}

			// Create line snippet description for approval message
			let lineSnippet = ""
			if (startLine !== undefined && endLine !== undefined) {
				lineSnippet = t("tools:readFile.linesRange", { start: startLine + 1, end: endLine + 1 })
			} else if (startLine !== undefined) {
				lineSnippet = t("tools:readFile.linesFromToEnd", { start: startLine + 1 })
			} else if (endLine !== undefined) {
				lineSnippet = t("tools:readFile.linesFromStartTo", { end: endLine + 1 })
			} else if (maxReadFileLine === 0) {
				lineSnippet = t("tools:readFile.definitionsOnly")
			} else if (maxReadFileLine > 0) {
				lineSnippet = t("tools:readFile.maxLines", { max: maxReadFileLine })
			}

			cline.consecutiveMistakeCount = 0
			const absolutePath = path.resolve(cline.cwd, relPath)

			const completeMessage = JSON.stringify({
				...sharedMessageProps,
				content: absolutePath,
				reason: lineSnippet,
			} satisfies ClineSayTool)

			console.log(`[readFileTool] Requesting approval for file read: ${relPath}`)
			const didApprove = await askApproval("tool", completeMessage)
			if (!didApprove) {
				console.log(`[readFileTool] File read approval denied for ${relPath}`)
				return
			}
			console.log(`[readFileTool] File read approval granted for ${relPath}`)

			// Count total lines in the file
			let totalLines = 0
			try {
				totalLines = await countFileLines(absolutePath)
			} catch (error) {
				console.error(`[readFileTool] Error counting lines in file ${absolutePath}:`, error)
			}

			// now execute the tool like normal
			let content: string
			let isFileTruncated = false
			let sourceCodeDef = ""

			console.log(`[readFileTool] Checking if file is binary: ${absolutePath}`)
			const isBinary = await isBinaryFile(absolutePath).catch((err) => {
				console.error(`[readFileTool] Error checking if file is binary:`, err)
				return false
			})
			console.log(`[readFileTool] File is binary: ${isBinary}`)

			if (isRangeRead) {
				console.log(
					`[readFileTool] Performing range read | Start line: ${startLine !== undefined ? startLine + 1 : "start"} | End line: ${endLine !== undefined ? endLine + 1 : "end"}`,
				)
				if (startLine === undefined) {
					content = addLineNumbers(await readLines(absolutePath, endLine, startLine))
				} else {
					content = addLineNumbers(await readLines(absolutePath, endLine, startLine), startLine + 1)
				}
			} else if (!isBinary && maxReadFileLine >= 0 && totalLines > maxReadFileLine) {
				// If file is too large, only read the first maxReadFileLine lines
				isFileTruncated = true
				console.log(
					`[readFileTool] File too large (${totalLines} lines), truncating to ${maxReadFileLine} lines`,
				)

				try {
					const res = await Promise.all([
						maxReadFileLine > 0 ? readLines(absolutePath, maxReadFileLine - 1, 0) : "",
						parseSourceCodeDefinitionsForFile(absolutePath, cline.rooIgnoreController),
					])

					content = res[0].length > 0 ? addLineNumbers(res[0]) : ""
					const result = res[1]
					if (result) {
						sourceCodeDef = `${result}`
						console.log(`[readFileTool] Successfully parsed source code definitions`)
					}
				} catch (err) {
					console.error(`[readFileTool] Error reading truncated file:`, err)
					throw err
				}
			} else {
				// Read entire file
				console.log(`[readFileTool] Reading entire file: ${absolutePath} | Total lines: ${totalLines}`)
				try {
					content = await extractTextFromFile(absolutePath)
					console.log(`[readFileTool] Successfully read entire file | Content length: ${content.length}`)
				} catch (err) {
					console.error(`[readFileTool] Error reading entire file:`, err)
					throw err
				}
			}

			// Create variables to store XML components
			let xmlInfo = ""
			let contentTag = ""

			// Add truncation notice if applicable
			if (isFileTruncated) {
				xmlInfo += `<notice>Showing only ${maxReadFileLine} of ${totalLines} total lines. Use start_line and end_line if you need to read more</notice>\n`

				// Add source code definitions if available
				if (sourceCodeDef) {
					xmlInfo += `<list_code_definition_names>${sourceCodeDef}</list_code_definition_names>\n`
				}
			}

			// Empty files (zero lines)
			if (content === "" && totalLines === 0) {
				// Always add self-closing content tag and notice for empty files
				contentTag = `<content/>`
				xmlInfo += `<notice>File is empty</notice>\n`
			}
			// Range reads should always show content regardless of maxReadFileLine
			else if (isRangeRead) {
				// Create content tag with line range information
				let lineRangeAttr = ""
				const displayStartLine = startLine !== undefined ? startLine + 1 : 1
				const displayEndLine = endLine !== undefined ? endLine + 1 : totalLines
				lineRangeAttr = ` lines="${displayStartLine}-${displayEndLine}"`

				// Maintain exact format expected by tests
				contentTag = `<content${lineRangeAttr}>\n${content}</content>\n`
			}
			// maxReadFileLine=0 for non-range reads
			else if (maxReadFileLine === 0) {
				// Skip content tag for maxReadFileLine=0 (definitions only mode)
				contentTag = ""
			}
			// Normal case: non-empty files with content (non-range reads)
			else {
				// For non-range reads, always show line range
				let lines = totalLines
				if (maxReadFileLine >= 0 && totalLines > maxReadFileLine) {
					lines = maxReadFileLine
				}
				const lineRangeAttr = ` lines="1-${lines}"`

				// Maintain exact format expected by tests
				contentTag = `<content${lineRangeAttr}>\n${content}</content>\n`
			}

			// Track file read operation
			if (relPath) {
				console.log(`[readFileTool] Tracking file context for ${relPath}`)
				await cline.getFileContextTracker().trackFileContext(relPath, "read_tool" as RecordSource)
			}

			// Log content length and estimated token count for diagnostics
			const contentLength = content ? content.length : 0
			console.log(`[readFileTool] Content length (chars): ${contentLength}`)
			// Simple token estimate: split on whitespace (not exact, but useful for diagnostics)
			const estimatedTokens = content ? content.split(/\s+/).length : 0
			console.log(`[readFileTool] Estimated token count: ${estimatedTokens}`)

			// Format the result into the required XML structure
			const xmlResult = `<file><path>${relPath}</path>\n${contentTag}${xmlInfo}</file>`
			pushToolResult(xmlResult)

			const endTime = Date.now()
			console.log(`[readFileTool] File read operation completed in ${endTime - startTime}ms | File: ${relPath}`)
		}
	} catch (error) {
		const errorMsg = error instanceof Error ? error.message : String(error)
		console.error(`[readFileTool] Error reading file ${relPath || ""}:`, error)

		// Track error in stats
		fileReadStats.errors.push({
			time: Date.now(),
			file: relPath || "",
			error: errorMsg,
		})

		// Log detailed error stats
		console.error(`[readFileTool] Error stats:`, {
			totalReads: fileReadStats.totalReads,
			totalErrors: fileReadStats.errors.length,
			consecutiveReads: fileReadStats.consecutiveReads,
			modelReads: fileReadStats.readsByModel.get(currentModelId) || 0,
			fileReads: relPath ? fileReadStats.readsByFile.get(relPath) || 0 : 0,
		})

		pushToolResult(`<file><path>${relPath || ""}</path><error>Error reading file: ${errorMsg}</error></file>`)
		await handleError("reading file", error)
	} finally {
		const endTime = Date.now()
		console.log(`[readFileTool] Operation completed in ${endTime - startTime}ms`)

		// Log summary if we've hit a high number of reads
		if (fileReadStats.totalReads % 5 === 0 || fileReadStats.consecutiveReads >= 10) {
			console.log(`[readFileTool] SUMMARY STATS:`, {
				totalReads: fileReadStats.totalReads,
				consecutiveReads: fileReadStats.consecutiveReads,
				totalErrors: fileReadStats.errors.length,
				recentErrors: fileReadStats.errors.slice(-3),
				topFiles: Array.from(fileReadStats.readsByFile.entries())
					.sort((a, b) => b[1] - a[1])
					.slice(0, 5),
				modelStats: Array.from(fileReadStats.readsByModel.entries()),
			})
		}
	}
}
